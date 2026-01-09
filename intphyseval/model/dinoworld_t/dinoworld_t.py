import torch
import torch.nn as nn
from torch import Tensor
import timm
from timm.models import Eva
from timm.models._manipulate import checkpoint
from jaxtyping import Float, Bool
from typing import Callable, Mapping
from einops import rearrange
from intphyseval.model.dino_world.dinoworld import (
    Block, 
    TimmViTEncoder,
    CrossAttentionPredictor,
    get_all_video_coords,
    next_frame_prediction_masks_compiled
)

class noop(nn.Module):
    def forward(self, x, **kwargs):
        return x

class GlobalSpatioTemporalBlock(nn.Module):
    """global (causal) spatio-temporal block"""

    def __init__(
        self,
        num_frames: int,
        grid_size: tuple[int, int],
        *,
        is_causal: bool,
        # Block kwargs
        dim: int,
        drop_path: float | int,
        norm_layer: Callable[[int], nn.Module],
        drop_path_type: str = "efficient",
        attn_kwargs: Mapping = {},
    ):
        super().__init__()
        self.num_frames = num_frames
        self.grid_size = grid_size
        self.num_patches = grid_size[0] * grid_size[1]
        self.is_causal = is_causal
        self.block = Block(
            dim=dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            context_dim=None,
            drop_path_type=drop_path_type,
            attn_kwargs=attn_kwargs,
        )
        self.attn_mask: Bool[Tensor, "T*N_patch T*N_patch"] | None
        if self.is_causal:
            t = torch.arange(num_frames).repeat_interleave(self.num_patches)
            attn_mask = t[None, :] <= t[:, None]
            self.register_buffer("attn_mask", attn_mask, persistent=False)
        else:
            self.attn_mask = None

    def forward(
        self,
        x: Float[Tensor, "B*T N D"],
        coords: Float[Tensor, "B T*N_patch 3"],
    ) -> Float[Tensor, "B*T N D"]:
        (BT, N, D), T = x.shape, self.num_frames
        N_pref, N_patch = N - self.num_patches, self.num_patches
        B = BT // T

        x_pref, x_patch = x[:, :N_pref], x[:, N_pref:]
        x_patch = x_patch.reshape(B, T * N_patch, D)
        x_patch = self.block(x_patch, coords=coords, attn_mask=self.attn_mask)
        x_patch = x_patch.reshape(B * T, N_patch, D)
        x = torch.cat([x_pref, x_patch], dim=1)
        return x


class TemporalViT(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_frames: int,
        t_indices: list[int],
        st_block: Callable[..., nn.Module],  # partial of a temporal block
        pretrained: bool = True,
        freeze_backbone: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.backbone: Eva = timm.create_model(
            model_name, pretrained=pretrained, **kwargs
        )
        self.num_frames = num_frames
        self.num_patches = self.backbone.patch_embed.num_patches
        self.grid_size = self.backbone.patch_embed.grid_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # check that all indices are valid (unique and in range)
        n_blocks = len(self.backbone.blocks)
        assert all(0 <= idx < n_blocks for idx in t_indices)
        assert len(t_indices) == len(set(t_indices))

        # create temporal modules
        self.t_blocks: list[nn.Module] = [noop() for _ in range(n_blocks)]
        for i in t_indices:
            self.t_blocks[i] = st_block(num_frames=num_frames, grid_size=self.grid_size)
        self.t_blocks = nn.ModuleList(self.t_blocks)
        self.grad_checkpointing = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing."""
        self.grad_checkpointing = enable

    def forward(
        self,
        x: Float[Tensor, "B T C h w"],  # h=height
        coords: Float[Tensor, "B T H W 3"],  # H=num of patches in height
    ) -> Float[Tensor, "B T H W D"]:
        assert not getattr(self.backbone, "rope_mixed", False)
        # assert not self.backbone.grad_checkpointing # We might enable GC, let's relax this assertion or check it
        assert x.shape[1] == self.num_frames

        # forward pass
        B, T, _, height, width = x.shape
        x = x.flatten(0, 1)  # (B*T, C, h, w)
        coords = coords.flatten(1, 3)  # (B, T*H*W, 3)
        # note: this flat order must match the one in patch embedding, assuming row-major
        x = self.backbone.patch_embed(x)
        x, rot_pos_embed = self.backbone._pos_embed(x)
        x = self.backbone.norm_pre(x)
        for blk, t_blk in zip(self.backbone.blocks, self.t_blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
                x = checkpoint(t_blk, x, coords=coords)
            else:
                x = blk(x, rope=rot_pos_embed)
                x = t_blk(x, coords=coords)

        x = self.backbone.norm(x)

        if self.backbone.num_prefix_tokens:
            x = x[:, self.backbone.num_prefix_tokens :]
        x = x.view(B, T, self.grid_size[0], self.grid_size[1], -1)
        return x
