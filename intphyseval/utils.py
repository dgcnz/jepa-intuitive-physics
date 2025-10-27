# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import logging
import sys
from torch.nn.functional import pad


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


PROPERTIES_BY_DATASET = {
    "intphys": ["O1", "O2", "O3"],
    "grasp": [
        "Collision",
        "Continuity",
        "Gravity",
        "GravityContinuity",
        "GravityInertia",
        "GravityInertia2",
        "GravitySupport",
        "Inertia",
        "Inertia2",
        "ObjectPermanence",
        "ObjectPermanence2",
        "ObjectPermanence3",
        "SolidityContinuity",
        "SolidityContinuity2",
        "Unchangeableness",
        "Unchangeableness2",
    ],
    "grasp_v2": [
        "Collision",
        "Continuity",
        "Gravity",
        "GravityContinuity",
        "GravityInertia",
        "GravityInertia2",
        "GravitySupport",
        "Inertia",
        "Inertia2",
        "ObjectPermanence",
        "ObjectPermanence2",
        "ObjectPermanence3",
        "SolidityContinuity",
        "SolidityContinuity2",
        "Unchangeableness",
        "Unchangeableness2",
    ],
    "inflevel_lab": [
        "continuity",
        "solidity",
        "gravity",
    ],
    "inflevel_lab_priming": [
        "continuity",
        "solidity",
        "gravity",
    ],
}


def get_time_masks(
    n_timesteps,
    spatial_size=(16, 16),
    temporal_size=2,
    spatial_dim=(224, 224),
    temporal_dim=16,
    as_bool=False,
):
    assert n_timesteps % temporal_size == 0
    x, y = spatial_dim
    t = temporal_dim

    num_patches_spatial = x / spatial_size[0] * x / spatial_size[0]
    num_patches_time = t / temporal_size
    patches_n_timesteps = int(num_patches_spatial * n_timesteps // temporal_size)

    patch_idcs = torch.arange(
        start=0, end=int(num_patches_spatial * num_patches_time), dtype=int
    )
    if as_bool:
        mask_enc = patch_idcs < patches_n_timesteps
        mask_pred = patch_idcs >= patches_n_timesteps

        full_mask = patch_idcs >= 0
    else:
        mask_enc = patch_idcs[:patches_n_timesteps]
        mask_pred = patch_idcs[patches_n_timesteps:]

        full_mask = patch_idcs

    return mask_enc, mask_pred, full_mask


def get_breaking_points(clip):
    bps = []
    for diff in [clip[0] - clip[1], clip[0] - clip[2], clip[0] - clip[3]]:
        try:
            i = torch.argwhere(diff.sum(2).sum(2).sum(0) != 0)[0, 0].item()
        except:
            i = clip.shape[2]
        bps.append(i)
    return bps


def get_matches(bps):
    if np.argmax(bps) == 0:
        return [[0, 1], [2, 3]]
    elif np.argmax(bps) == 1:
        return [[0, 2], [1, 3]]
    else:
        return [[0, 3], [1, 2]]


def get_action_timestep(matched_clips):
    diff = matched_clips[0] - matched_clips[1]
    return torch.argwhere(diff.sum(2).sum(2).sum(0) != 0)[0, 0].item()


def pad_tensors(tensors, max_length, length_axis=-1):
    padded_tensors = []
    for t in tensors:
        padding_needed = max_length - t.size(length_axis)
        padded_tensors.append(pad(t, (0, padding_needed)))
    return padded_tensors
