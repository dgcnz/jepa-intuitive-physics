# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pathlib
import warnings

from logging import getLogger

import numpy as np
import pandas as pd

from PIL import Image
import torch
import json


_GLOBAL_SEED = 0
logger = getLogger()


def make_videodataset(
    data_path,
    property,
    batch_size,
    frames_per_clip=8,
    frame_step=4,
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    deterministic=True,
    log_dir=None,
    priming=False,
):
    dataset = IntPhysDataset(
        data_path=os.path.join(data_path, property),
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        shared_transform=shared_transform,
        transform=transform,
    )

    logger.info("Dataset created")
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=drop_last
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=True,
    )

    logger.info("Intuitive Physics data loader created")

    return dataset, data_loader, dist_sampler


class IntPhysDataset(torch.utils.data.Dataset):
    """IntPhys dataset"""

    def __init__(
        self,
        data_path,
        frames_per_clip=16,
        frame_step=4,
        transform=None,
        shared_transform=None,
        random_start: bool = False,
    ):
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.transform = transform
        self.shared_transform = shared_transform

        self.scenes = sorted(os.listdir(self.data_path))
        self.length_clip = self.frames_per_clip * self.frame_step
        # Random start makes alignment between clips harder and
        # with the default setting it's not even applied:
        # * frame_step      = 2
        # * frames_per_clip = 99//frame_step
        # => length_clip    = 98
        self.random_start = random_start
        if self.random_start:
            assert self.length_clip <= 90, (
                "It makes no sense to have random start with full clips"
            )

    def __getitem__(self, index):
        scene = self.scenes[index]
        # To change if  we want to get a precise one/all of them
        labels = []
        buffer = []
        paths = []
        for possibility in [1, 2, 3, 4]:
            frames_all = sorted(
                os.listdir(f"{self.data_path}/{scene}/{possibility}/scene")
            )

            if self.random_start:
                start = np.random.randint(0, len(frames_all) - self.length_clip)
            else:
                start = 0
            frames_to_load = frames_all[
                start : start + self.length_clip : self.frame_step
            ]

            # get
            clip = []
            for frame in frames_to_load:
                frame_ = Image.open(
                    f"{self.data_path}/{scene}/{possibility}/scene/{frame}"
                )
                frame_ = torch.Tensor(np.array(frame_))
                clip.append(frame_)
            clip = torch.stack(clip)
            if self.transform is not None:
                clip = self.transform(clip)
            buffer.append(clip)
            try:
                with open(
                    f"{self.data_path}/{scene}/{possibility}/status.json", "r"
                ) as f:
                    a = json.load(f)
                label = a["header"]["is_possible"]
            except:
                label = True
            labels.append(label)

            paths.append(f"{self.data_path[-2:]}/{scene}/{possibility}")
        buffer = torch.stack(buffer)
        labels = torch.Tensor(labels)

        return buffer, labels

    def __len__(self):
        return len(self.scenes)
