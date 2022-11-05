from contextlib import suppress
import warnings
from functools import partial
from easy_transformer import EasyTransformer
import plotly.graph_objects as go
import numpy as np
from numpy import sin, cos, pi
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from tqdm import tqdm
import pandas as pd
import torch
import plotly.express as px
import gc
import einops
from easy_transformer.experiments import get_act_hook
import time


def ctime2():
    return time.ctime().replace(" ", "_")


def get_hook(layer, head_idx):
    if head_idx is None:
        return (f"blocks.{layer}.hook_mlp_out", None)
    else:
        return (f"blocks.{layer}.attn.hook_result", head_idx)


def get_number_in_string(string):
    return int("".join(filter(str.isdigit, string)))


def get_all_subsets(my_list):
    if len(my_list) == 0:
        return [[]]
    subsets = get_all_subsets(my_list[1:])
    return subsets + [[my_list[0]] + s for s in subsets]


def prepend_padding(tens, model_tokenizer=None, pad_token=None):
    """
    Add a padding token
    """

    assert (
        model_tokenizer is not None or pad_token is not None
    ), "Either model_tokenizer or pad_token must be provided"

    if pad_token is None:
        pad_token = model_tokenizer.pad_token_id

    assert len(tens.shape) == 2, f"{tens.shape} not 2D"

    new_tens = torch.zeros((tens.shape[0], tens.shape[1] + 1), dtype=tens.dtype)
    new_tens[:, 1:] = tens
    new_tens[:, 0] = pad_token
    return new_tens


def patch_all(z, source_act, hook):
    # z[start_token:end_token] = source_act[start_token:end_token]
    z = source_act
    # z[:] = 0
    return z


def loss_metric(
    model,
    rand_tokens_repeat,
    seq_len,
):
    cur_loss = model(
        rand_tokens_repeat, return_type="both", loss_return_per_token=True
    )["loss"][:, -seq_len // 2 :].mean()
    return cur_loss.item()


def logits_metric(
    model,
    rand_tokens_repeat,
    # seq_len,
):
    assert len(rand_tokens_repeat.shape) == 2
    assert rand_tokens_repeat.shape[1] % 2 == 1
    seq_len = rand_tokens_repeat.shape[1] // 2
    logits = model(rand_tokens_repeat, return_type="logits")
    # print(logits.shape) # 5 21 50257

    assert len(logits.shape) == 3, logits.shape
    batch_size, _, vocab_size = logits.shape
    seq_indices = einops.repeat(
        torch.arange(seq_len) + seq_len, "a -> b a", b=batch_size
    )
    batch_indices = einops.repeat(torch.arange(batch_size), "b -> b a", a=seq_len)
    logits_on_correct = logits[
        batch_indices, seq_indices, rand_tokens_repeat[:, seq_len + 1 :]
    ]

    return logits_on_correct[:, -seq_len // 2 :].mean().item()


def denom_metric(
    model,
    rand_tokens_repeat,
    seq_len,
):
    """Denom of the final softmax"""
    logits = model(rand_tokens_repeat, return_type="logits")  # 5 21 50257
    denom = torch.exp(logits)
    denom = denom[:, -seq_len // 2 :].sum(dim=-1).mean()
    return denom.item()


def path_patching_attribution(
    model,
    tokens,
    patch_tokens,
    sender_heads,
    receiver_hooks,
    verbose=False,
    max_layer=12,
    return_hooks=False,
    extra_hooks=[],  # when we call reset hooks, we may want to add some extra hooks after this, add these here
    freeze_mlps=False,  # recall in IOI paper we consider these "vital model components"
    have_internal_interactions=False,
    device="cuda",
    do_assert=True,
    **kwargs,
):
    """
    Do path patching in order to see which heads matter the most
    for directly writing the correct answer (see loss change)
    """

    if len(kwargs) > 0:
        warnings.warn("KWARGS snuck in: " + str(kwargs.keys()))
    if not freeze_mlps:
        warnings.warn("Not freezing MLPs")
    for hook in receiver_hooks:
        if get_number_in_string(hook[0]) < max_layer:
            warnings.warn("Is max layer too large?")

    # see path patching in ioi utils
    sender_hooks = []

    for layer, head_idx in sender_heads:
        if head_idx is None:
            sender_hooks.append((f"blocks.{layer}.hook_mlp_out", None))

        else:
            sender_hooks.append((f"blocks.{layer}.attn.hook_result", head_idx))

    sender_hook_names = [x[0] for x in sender_hooks]
    receiver_hook_names = [x[0] for x in receiver_hooks]

    sender_cache = {}
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    model.cache_some(
        sender_cache,
        lambda x: x in sender_hook_names,
        suppress_warning=True,
        device=device,
    )
    source_logits, source_loss = model(
        patch_tokens, return_type="both", loss_return_per_token=True
    ).values()

    target_cache = {}
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    model.cache_some(
        target_cache,
        lambda x: (
            "attn.hook_q" in x
            or "hook_mlp_out" in x
            or "attn.hook_k" in x
            or "attn.hook_v" in x
            or "attn.hook_result" in x  # TODO delete, for debugging
        ),
        suppress_warning=True,
        device=device,
    )
    target_logits, target_loss = model(
        tokens, return_type="both", loss_return_per_token=True
    ).values()

    # measure the receiver heads' values
    receiver_cache = {}
    model.reset_hooks()
    model.cache_some(
        receiver_cache,
        lambda x: x in receiver_hook_names,
        suppress_warning=True,
        verbose=False,
        device=device,
    )

    # for all the Q, K, V things
    for layer in range(max_layer):  # make sure to ablate stuff before
        for head_idx in range(model.cfg.n_heads):
            for hook_template in [
                "blocks.{}.attn.hook_q",
                "blocks.{}.attn.hook_k",
                "blocks.{}.attn.hook_v",
            ]:
                hook_name = hook_template.format(layer)

                if have_internal_interactions and hook_name in receiver_hook_names:
                    continue

                hook = get_act_hook(
                    patch_all,
                    alt_act=target_cache[hook_name],
                    idx=head_idx,
                    dim=2 if head_idx is not None else None,
                    name=hook_name,
                )
                model.add_hook(hook_name, hook)

        if freeze_mlps:
            hook_name = f"blocks.{layer}.hook_mlp_out"
            hook = get_act_hook(
                patch_all,
                alt_act=target_cache[hook_name],
                idx=None,
                dim=None,
                name=hook_name,
            )
            model.add_hook(hook_name, hook)

    for hook in extra_hooks:
        # ughhh, think that this is what we want, this should override the QKV above
        model.add_hook(*hook)

    # we can override the hooks above for the sender heads, though
    for hook_name, head_idx in sender_hooks:
        if do_assert:
            assert not torch.allclose(
                target_cache[hook_name],
                sender_cache[hook_name],
            )

        hook = get_act_hook(
            patch_all,
            alt_act=sender_cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        )
        model.add_hook(hook_name, hook)

    receiver_logits, receiver_loss = model(
        tokens, return_type="both", loss_return_per_token=True
    ).values()

    # receiver_cache stuff ...
    # patch these values in
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(
            *hook
        )  # ehh probably doesn't actually matter cos end thing hooked

    hooks = []
    for hook_name, head_idx in receiver_hooks:
        # if torch.allclose(receiver_cache[hook_name], target_cache[hook_name]):
        #     assert False, (hook_name, head_idx)
        hook = get_act_hook(
            patch_all,
            alt_act=receiver_cache[hook_name].clone(),
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        )
        hooks.append((hook_name, hook))

    for obj in ["receiver_cache", "target_cache", "sender_cache"]:
        e(obj)

    model.reset_hooks()
    if return_hooks:
        return hooks
    else:
        for hook_name, hook in hooks:
            model.add_hook(hook_name, hook)
        return model


def ppa_multiple(
    model,
    tokens,
    patch_tokens,
    attention_max_layer,
    mlp_max_layer,
    receiver_hooks,
    metric,
    # pos, # TODO for now we YOLO this
):
    """
    Do a bunch of path patching experiments (e.g getting the heatmap)
    """

    head_results = torch.zeros(size=(model.cfg.n_layers, model.cfg.n_heads))
    mlp_results = torch.zeros(size=(model.cfg.n_layers, 1))

    model.reset_hooks()
    initial_metric = metric(model, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head_idx in [None] + list(range(model.cfg.n_heads)):
            path_patching_attribution(
                model=model,
                tokens=tokens,
                patch_tokens=patch_tokens,
                sender_heads=[(layer, head_idx)],
                receiver_hooks=receiver_hooks,
                device="cuda",
                freeze_mlps=True,
                return_hooks=False,
                max_layer=11,
            )

            cur_metric = metric(model, tokens)
            if head_idx is None:
                mlp_results[layer] = cur_metric - initial_metric
            else:
                head_results[layer, head_idx] = cur_metric - initial_metric

    return head_results, mlp_results, initial_metric


def show_losses(
    models,
    model_names,
    rand_tokens_repeat,
    seq_len,
    mode="loss",
):
    batch_size = rand_tokens_repeat.shape[0]
    ys = [[] for _ in range(len(model_names))]
    fig = go.Figure()
    for idx, model_name in enumerate(model_names):
        model = models[idx]
        rand_tokens_repeat[:, 0] = model.tokenizer.bos_token_id
        logits, loss = model(
            rand_tokens_repeat, return_type="both", loss_return_per_token=True
        ).values()
        print(
            model_name,
            "loss and std on last quarter of token (TODO make this all relevant induction cases?)",
            loss[:, -seq_len // 2 :].mean().item(),
            loss[:, -seq_len // 2 :].std().item(),
        )

        if mode != "logits":
            mean_loss = loss.mean(dim=0)
            ys[idx] = mean_loss.detach().cpu()  # .numpy()

            if mode == "probs":
                ys[idx] = torch.exp(-ys[idx])
        else:
            # fairly cursed indexing ...
            assert len(logits.shape) == 3, logits.shape

            seq_indices = einops.repeat(
                torch.arange(seq_len * 2), "a -> b a", b=batch_size
            )
            batch_indices = einops.repeat(
                torch.arange(batch_size), "b -> b a", a=seq_len * 2
            )

            logits_on_correct = logits[
                batch_indices, seq_indices, rand_tokens_repeat[:, 1:]
            ]

            ys[idx] = logits_on_correct.mean(dim=0).detach().cpu()

        print(ys[idx].shape)
        fig.add_trace(
            go.Scatter(
                y=ys[
                    idx
                ],  # torch.exp(-mean_loss.detach().cpu()) if mode == "probs" else mean_loss.detach().cpu(),
                name=model_name,
                mode="lines",
                # line=dict(color=CLASS_COLORS[idx]),
            )
        )
    fig.update_layout(title=f"{mode} over time")

    # add a line at x = 50 saying that this should be the first guessable
    fig.add_shape(
        type="line",
        x0=seq_len,
        y0=0,
        x1=seq_len,
        y1=ys[0].max(),
        line=dict(color="Black", width=1, dash="dash"),
    )
    # add a label to this line
    fig.add_annotation(
        x=seq_len,
        y=ys[0].max(),
        text="First case of induction",
        showarrow=False,
        font=dict(size=16),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=0,
        ay=-seq_len - 5,
    )

    fig.show()


def get_position(my_list):
    """
    Given a list l, return the position of l[0] in l when sorted
    """
    first = my_list[0]
    my_list = sorted(my_list)
    return my_list.index(first)
