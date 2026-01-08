import torch
import torch.nn as nn
import logging
import sys
import types
import numpy as np
from intphyseval.model.dino_world.dinoworld import (
    TimmViTEncoder,
    CrossAttentionPredictor,
    get_all_video_coords,
    _init_weights,
)
from functools import partial

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class DinoWorld(nn.Module):
    def __init__(
        self,
        encoder: TimmViTEncoder,
        predictor: CrossAttentionPredictor,
        num_frames: int,
        img_size: int,
        patch_size: int,
        tubelet_size: int = 1,
        fps: float = 4.0,
        normalize_targets: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.fps = fps
        self.normalize_targets = normalize_targets

    def forward(self, pieces, masks_enc, masks_pred, full_mask):
        pieces = pieces.permute(0, 2, 1, 3, 4)
        B, T, C, H, W = pieces.shape

        all_tokens = self.encoder(pieces)
        B, T, h_grid, w_grid, D = all_tokens.shape
        N_all = T * h_grid * w_grid

        all_tokens_flat = all_tokens.flatten(1, 3)

        timestamps = (
            torch.arange(T, device=pieces.device, dtype=pieces.dtype) / self.fps
        ).expand(B, T)

        coords = get_all_video_coords(timestamps, h_grid, w_grid)
        coords_flat = coords.flatten(1, 3)

        targets = all_tokens_flat[masks_pred].view(B, -1, D)
        coords_query = coords_flat[masks_pred].view(B, -1, 3)

        N_pred = targets.shape[1]
        ctx_mask = masks_enc.unsqueeze(1).expand(B, N_pred, N_all)

        preds = self.predictor(
            tokens=all_tokens,
            coords=coords_flat,
            ctx_mask=ctx_mask,
            coords_query=coords_query,
        )

        if self.normalize_targets:
            targets = torch.nn.functional.layer_norm(targets, (targets.size(-1),))

        return preds, targets

    def freeze(self):
        self.encoder.eval()
        self.predictor.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.predictor.parameters():
            p.requires_grad = False

    def load_ckpt(self, ckpt: str):
        logger.info(f"Loading DinoWorld checkpoint from {ckpt}")

        for n in ("src", "src.utils", "src.utils.scheduler"):
            sys.modules.setdefault(n, types.ModuleType(n))
            sys.modules["src.utils.scheduler"].CAPIScheduler = type(
                "CAPIScheduler", (), {}
            )

        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"]

        def load_component(component, prefix, state_dict):
            comp_sd = {
                k[len(prefix) + 1 :]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if not comp_sd:
                return

            missing, unexpected = component.load_state_dict(comp_sd, strict=False)
            logger.info(
                f"Loaded {prefix}. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
            )

        load_component(self.encoder, "encoder", state_dict)
        load_component(self.predictor, "predictor", state_dict)

        return self
