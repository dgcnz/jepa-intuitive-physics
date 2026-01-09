import logging
import sys
import types
import torch
import torch.nn as nn
from intphyseval.model.dinoworld_t.dinoworld_t import (
    TemporalViT,
    TimmViTEncoder,
    CrossAttentionPredictor,
    get_all_video_coords,
)

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class DinoWorldT(nn.Module):
    def __init__(
        self,
        encoder: TemporalViT,
        target_encoder: TimmViTEncoder,
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
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.fps = fps
        self.normalize_targets = normalize_targets

    def forward(self, pieces, masks_enc, masks_pred, full_mask):
        pieces = pieces.permute(0, 2, 1, 3, 4)
        B, T, _, _, _ = pieces.shape

        with torch.no_grad():
            frame_gt_tokens = self.target_encoder(pieces)

        _, _, h_grid, w_grid, D = frame_gt_tokens.shape
        N_all = T * h_grid * w_grid

        timestamps = (
            torch.arange(T, device=pieces.device, dtype=pieces.dtype) / self.fps
        ).expand(B, T)

        coords = get_all_video_coords(timestamps, h_grid, w_grid)

        frame_tokens = self.encoder(pieces, coords)

        frame_gt_tokens_flat = frame_gt_tokens.flatten(1, 3)
        coords_flat = coords.flatten(1, 3)

        targets = frame_gt_tokens_flat[masks_pred].view(B, -1, D)
        coords_query = coords_flat[masks_pred].view(B, -1, 3)

        N_pred = targets.shape[1]
        ctx_mask = masks_enc.unsqueeze(1).expand(B, N_pred, N_all)

        preds = self.predictor(
            tokens=frame_tokens,
            coords=coords_flat,
            ctx_mask=ctx_mask,
            coords_query=coords_query,
        )

        if self.normalize_targets:
            targets = torch.nn.functional.layer_norm(targets, (targets.size(-1),))

        return preds, targets

    def freeze(self):
        self.encoder.eval()
        self.target_encoder.eval()
        self.predictor.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.predictor.parameters():
            p.requires_grad = False

    def load_ckpt(self, ckpt: str):
        logger.info(f"Loading DinoWorldT checkpoint from {ckpt}")
        sys.modules.setdefault("src", types.ModuleType("src")).__path__ = []
        sys.modules.setdefault("src.utils", types.ModuleType("src.utils")).__path__ = []

        sched = sys.modules.setdefault(
            "src.utils.scheduler", types.ModuleType("src.utils.scheduler")
        )
        sched.__getattr__ = lambda name, _c={}: _c.setdefault(name, type(name, (), {}))

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
        load_component(self.target_encoder, "target_encoder", state_dict)
        load_component(self.predictor, "predictor", state_dict)

        return self
