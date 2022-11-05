# %%
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from utils_induction import (
    prepend_padding,
    logits_metric,
    loss_metric,
    path_patching_attribution,
)
import torch
import einops
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from ioi_dataset import IOIDataset

from easy_transformer import EasyTransformer

from ioi_dataset import (
    IOIDataset,
)
from utils_circuit_discovery import (
    path_patching,
    logit_diff_io_s,
    HypothesisTree,
    show_pp,
)

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
# %%
model_name = "EleutherAI/gpt-neo-125M"  # @param ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'facebook/opt-125m', 'facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b']

model = EasyTransformer.from_pretrained(model_name)
if torch.cuda.is_available():
    model.to("cuda")
model.set_use_attn_result(True)

# %%
orig = "When John and Mary went to the store, John gave a bottle of milk to Mary."
new = "When John and Mary went to the store, Charlie gave a bottle of milk to Mary."
# new = "A completely different gibberish sentence blalablabladfghjkoiuytrdfg"

model.reset_hooks()
logit = model(orig)[0, 16, 5335]

model = path_patching(
    model,
    orig,
    new,
    [(5, 5)],
    [("blocks.8.attn.hook_v", 6)],
    12,
    position=torch.tensor([16]),
)

new_logit = model(orig)[0, 16, 5335]
model.reset_hooks()
print(logit, new_logit)

# %%
N = 50
ioi_dataset = IOIDataset(
    prompt_type="ABBA",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
)
abc_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)
# %%
# the circuit discovery algorithm
#
# we want to write down the process of discovery the IOI circuit as an algorithm
# 1. start at the logits at the token position 'end', run path patching on each head and mlp.
# 2. pick threshold (probably in terms of percentage change in metric we care about?), identify components that have effect sizes above threshold
# 3. for comp in identified components:
## a. run path patching on all components upstream to it, with the q, k, or v part of comp as receiver
#%% [markdown]
# Main part of the automatic circuit discovery algorithm


positions = OrderedDict()
positions["IO"] = ioi_dataset.word_idx["IO"]
positions["S"] = ioi_dataset.word_idx["S"]
positions["S+1"] = ioi_dataset.word_idx["S+1"]
positions["S2"] = ioi_dataset.word_idx["S2"]
positions["end"] = ioi_dataset.word_idx["end"]

h = HypothesisTree(
    model,
    metric=logit_diff_io_s,
    dataset=ioi_dataset,
    orig_data=ioi_dataset.toks.long(),
    new_data=abc_dataset.toks.long(),
    threshold=0.2,
    possible_positions=positions,
    use_caching=True,
)

# %%
h.eval(show_graphics=False)
while h.current_node is not None:
    h.eval(verbose=True, show_graphics=False, auto_threshold=True)
    h.show(save=True)
h.show(save=True)

# %%
attn_results_fast = deepcopy(h.attn_results)
mlp_results_fast = deepcopy(h.mlp_results)

#%% [markdown]
# Make induction dataset

seq_len = 10
batch_size = 5
interweave = 10  # have this many things before a repeat

rand_tokens = torch.randint(1000, 10000, (batch_size, seq_len))
rand_tokens_repeat = torch.zeros(
    size=(batch_size, seq_len * 2)
).long()  # einops.repeat(rand_tokens, "batch pos -> batch (2 pos)")

for i in range(seq_len // interweave):
    rand_tokens_repeat[
        :, i * (2 * interweave) : i * (2 * interweave) + interweave
    ] = rand_tokens[:, i * interweave : i * interweave + interweave]
    rand_tokens_repeat[
        :, i * (2 * interweave) + interweave : i * (2 * interweave) + 2 * interweave
    ] = rand_tokens[:, i * interweave : i * interweave + interweave]
rand_tokens_control = torch.randint(1000, 10000, (batch_size, seq_len * 2))

rand_tokens = prepend_padding(rand_tokens, model.tokenizer)
rand_tokens_repeat = prepend_padding(rand_tokens_repeat, model.tokenizer)
rand_tokens_control = prepend_padding(rand_tokens_control, model.tokenizer)


def calc_score(attn_pattern, hook, offset, arr):
    # Pattern has shape [batch, index, query_pos, key_pos]
    stripe = attn_pattern.diagonal(offset, dim1=-2, dim2=-1)
    scores = einops.reduce(stripe, "batch index pos -> index", "mean")
    # Store the scores in a common array
    arr[hook.layer()] = scores.detach().cpu().numpy()
    # return arr
    return attn_pattern


def filter_attn_hooks(hook_name):
    split_name = hook_name.split(".")
    return split_name[-1] == "hook_attn"


#%% [markdown]
# is the model sane at induction?

model.reset_hooks()
initial_result = logits_metric(model, rand_tokens_repeat)
assert 14 <= initial_result <= 18, initial_result

#%% [markdown]

positions = OrderedDict()
batch_size, seq_len = rand_tokens_repeat.shape

for i in range(seq_len - 1):
    positions[str(i)] = torch.ones(size=(batch_size,)).long() * i

model.reset_hooks()
h = HypothesisTree(
    model,
    metric=logits_metric,
    dataset=rand_tokens_repeat,
    orig_data=rand_tokens_repeat,
    new_data=rand_tokens_control,
    threshold=0.05,  # percentage, I think
    possible_positions=positions,
    use_caching=True,
    loss_relevant_positions=[str(n) for n in range(15, 20)],
)

eval_round = 0
# while len(h.node_stack) > 0:
for _ in range(30): # should be enough!!!
    print(f"Evaluating round {eval_round}")
    eval_round += 1
    h.eval()
h.show() # should then be sad and cover up things : (

#%%

import matplotlib.pyplot as plt
import networkx as nx


def show(h, save=False, current_node=None):
    edge_color_list = []
    color_dict = {
        "q": "black",
        "k": "blue",
        "v": "green",
        "out": "red",
        "post": "red",
    }
    if current_node is None:
        current_node = h.root_node
    G = nx.DiGraph()

    def dfs(node):
        G.add_nodes_from([(node, {"layer":
         node.layer})])
        for child_node, child_score, child_type in node.parents:
            G.add_edges_from([(node, child_node, {"weight": round(child_score, 3)})])
            edge_color_list.append(color_dict[child_type])
            dfs(child_node)

    dfs(current_node)
    pos = nx.multipartite_layout(G, subset_key="layer")
    # make plt figure fills screen
    fig = plt.figure(dpi=300, figsize=(24, 24))
    nx.draw(
        G,
        pos,
        node_size=8000,
        node_color="#b0a8a7",
        linewidths=2.0,
        edge_color=edge_color_list,
        width=1.5,
        arrowsize=12,
    )

    nx.draw_networkx_labels(G, pos, font_size=14)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    if save:
        plt.savefig("ioi_circuit.png")

    plt.show()

# %%
