# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn


logger = logging.getLogger("dinov2")


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    
    # todo delete the following
    # print(pretrained_weights)
    # state_dict = torch.load(pretrained_weights, map_location="cpu")
    # print(state_dict.keys())
    # # print(state_dict['pos_embed'].shape)
    # print(state_dict['patch_embed.proj.weight'].shape)
    # # print(state_dict['patch_embed.proj.bias'].shape)
    # state_dict['patch_embed.proj.weight'] = nn.functional.interpolate(state_dict['patch_embed.proj.weight'], size=(16,16), mode='bilinear', align_corners=True) # [1,C,H,W]
    # print(state_dict['patch_embed.proj.weight'].shape)
    # torch.save(state_dict,'models/dinoV2/dinov2_vitl16_pretrain.pth')
    
    
    # # todo delete the following
    # print(pretrained_weights)
    # state_dict = torch.load('ECCV_record/data/2024.1.12/DimerUni/version_3/epoch=47-loss_epoch=2.4483-valid_epoch_EPE=0.5793.ckpt', map_location="cpu")
    # print(state_dict['state_dict']['model.Dimer.DinoV2.patch_embed.proj.weight'].shape)
    # state_dict['state_dict']['model.Dimer.DinoV2.patch_embed.proj.weight'] = nn.functional.interpolate(state_dict['state_dict']['model.Dimer.DinoV2.patch_embed.proj.weight'], size=(16,16), mode='bilinear', align_corners=True) # [1,C,H,W]
    # print(state_dict['state_dict']['model.Dimer.DinoV2.patch_embed.proj.weight'].shape)
    # torch.save(state_dict,'models/dinoV2/dinov2_vitl16_ECCV.ckpt')
    # state_dict = torch.load('models/dinoV2/dinov2_vitl16_ECCV.ckpt', map_location="cpu")
    # print(state_dict['state_dict']['model.Dimer.DinoV2.patch_embed.proj.weight'].shape)
    
    
    
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu",weights_only=True)


    # del_list = []
    # for k, v in state_dict.items():
    #     if 'depth_head' in k:
    #         del_list.append(k)
    # for i in del_list:
    #     del state_dict[i]
    # state_dict = {k.replace("pretrained.", ""): v for k, v in state_dict.items()}
    # torch.save(state_dict,'../toolkit/models/DepthAny_body_vitl.pth')
    # exit()    
    
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
