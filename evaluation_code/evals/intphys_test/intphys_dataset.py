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
):
    dataset = IntPhysDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        shared_transform=shared_transform,
        transform=transform)


    logger.info('Dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=drop_last)


    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=True)

    logger.info('Intuitive Physics data loader created')

    return dataset, data_loader, dist_sampler


def path_from_task(data_root,task):
        return data_root + task[:7]+task[-2:]
    
class IntPhysDataset(torch.utils.data.Dataset):
    """ IntPhys dataset """

    def __init__(
        self,
        data_path,
        frames_per_clip=16,
        frame_step=4,
        transform=None,
        shared_transform=None,
    ):
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.transform = transform
        self.shared_transform = shared_transform

        task_path = self.data_path + '../task.txt'
        with open(task_path,'r') as f:
            self.tasks = [line.strip() for line in f.readlines()]
            
        self.length_clip = self.frames_per_clip*self.frame_step


    def __getitem__(self, index):
        task = self.tasks[index]
        video_path  = path_from_task(self.data_path,task)
        # To change if  we want to get a precise one/all of them
        frames_all = sorted(os.listdir(f"{video_path}/scene"))

        start = np.random.randint(0,len(frames_all)-self.length_clip)
        if self.length_clip > 90 :
            start = 0
        frames_to_load = frames_all[start:start+self.length_clip:self.frame_step]

        #get
        clip = []
        for frame in frames_to_load:
            frame_ = Image.open(f"{video_path}/scene/{frame}")
            frame_ = torch.Tensor(np.array(frame_))
            clip.append(frame_)
        clip = torch.stack(clip)
        if self.transform is not None:
            clip = self.transform(clip)

        return clip, torch.Tensor([index])



    def __len__(self):
        return len(self.tasks)
