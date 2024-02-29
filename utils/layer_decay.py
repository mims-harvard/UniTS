# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 2

    layer_scales = list(layer_decay ** (num_layers - i)
                        for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_model(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
        print("name: %s, layer_id: %d, group_name: %s, lr_scale: %f, weight_decay: %f" % (
            n, layer_id, group_name, param_group_names[group_name]["lr_scale"], param_group_names[group_name]["weight_decay"]))

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id_for_model(name, num_layers):
    """
    Assign a parameter with its layer id
    """
    if name in ['cls_token', 'patch_embeddings', 'position_embedding', 'prompt2forecat', 'prompt_tokens', 'mask_tokens', 'cls_tokens', 'category_tokens']:
        return 0
    elif name.startswith('input_encoders'):
        return 1
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 2
    else:
        return num_layers
