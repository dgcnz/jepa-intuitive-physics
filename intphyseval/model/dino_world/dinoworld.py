import timm
from timm.models import VisionTransformer
import torch
from torch import Tensor, nn
from jaxtyping import Bool, Float
from collections.abc import Callable, Mapping
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial


class TimmViTEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, **kwargs) -> None:
        """
        :param model_name: Name of the timm model to load.
        :param pretrained: Whether to load pretrained weights.
        """
        super().__init__()
        self.backbone: VisionTransformer = timm.create_model(
            model_name,
            pretrained=pretrained,
            **kwargs,
        )
        self.embed_dim: int = self.backbone.embed_dim
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def forward(
        self,
        frames: Float[Tensor, "b t 3 H W"],
    ) -> Float[Tensor, "b t h w d"]:
        b, t, _, _, _ = frames.shape

        feats = self.backbone.forward_intermediates(
            frames.flatten(0, 1),
            indices=[-1],
            return_prefix_tokens=False,
            output_fmt="NCHW",
            intermediates_only=True,
            norm=True,
        )[0]
        _, d, h_p, w_p = feats.shape
        tokens = feats.permute(0, 2, 3, 1).contiguous()
        tokens = tokens.view(b, t, h_p, w_p, d)
        return tokens


def get_all_video_coords(
    times: Float[Tensor, "B T"],
    height: int,
    width: int,
) -> Float[Tensor, "B T H W 3"]:
    """
    Build (time, y, x) coordinates.
    Notes:
    - (y, x) are normalized to [-1, 1]
    - time is kept in the same unit as `times`

    :param times: Timestamps for each frame in seconds.
    :param height: Patch grid height.
    :param width: Patch grid width.
    """
    device, dtype = times.device, times.dtype
    ys = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    yy, xx = yy[None, None], xx[None, None]  # add (b, t) dims
    tau = times[..., None, None].expand(-1, -1, height, width)
    coords = torch.stack([tau, yy.expand_as(tau), xx.expand_as(tau)], dim=-1)
    return coords


def next_frame_prediction_masks_compiled(
    coords: Float[Tensor, "b t h w 3"],
) -> tuple[Bool[Tensor, "b n"], Bool[Tensor, "b n_pred n"]]:
    """
    Compile-friendly version of next_frame_prediction_masks.
    Uses slicing instead of boolean indexing where possible.
    """
    b, t, h, w, _ = coords.shape
    n_tokens = t * h * w
    num_patches = h * w

    # Predict all tokens of frames 1..T-1.
    frame_mask = torch.zeros((b, t), dtype=torch.bool, device=coords.device)
    frame_mask[:, 1:] = True
    pred_mask = frame_mask.repeat_interleave(num_patches, dim=1)  # (B, N)

    tau_all = coords[..., 0].view(b, n_tokens)  # (B, N)

    # Use slicing instead of boolean indexing
    tau_query = coords[:, 1:, ..., 0].flatten(1)
    n_pred = tau_query.shape[1]

    ctx = tau_all.unsqueeze(1).expand(-1, n_pred, -1)
    tgt = tau_query.unsqueeze(2).expand(-1, -1, n_tokens)
    ctx_mask = ctx < tgt

    return pred_mask, ctx_mask


# copied from CAPI
class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        mlp_ratio: int | float | None = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_features is None:
            assert mlp_ratio is not None
            hidden_features = int(in_features * mlp_ratio)
        else:
            assert mlp_ratio is None
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: Float[Tensor, "*b d"]) -> Float[Tensor, "*b d"]:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# copied from CAPI
class NaiveResidual(nn.Module):
    def __init__(self, drop_prob: float | int, norm: nn.Module, fn: nn.Module):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.keep_prob = 1 - drop_prob

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        **kwargs: Float[Tensor, "b ..."] | None,
    ) -> Float[Tensor, "b n d"]:
        fn_out = self.fn(self.norm(x), **kwargs)
        if self.keep_prob == 1.0 or not self.training:
            return x + fn_out
        mask = fn_out.new_empty(x.shape[0]).bernoulli_(self.keep_prob)[:, None, None]
        return x + fn_out * mask / self.keep_prob


# copied from CAPI
class EfficientResidual(NaiveResidual):
    def forward(
        self,
        x: Float[Tensor, "b n_q d"],
        attn_mask: Bool[Tensor, "n_q n_k"] | None = None,
        **kwargs: Float[Tensor, "b ..."] | None,
    ) -> Float[Tensor, "b n_q d"]:
        if attn_mask is not None:
            kwargs["attn_mask"] = attn_mask
        if self.keep_prob == 1.0 or not self.training:
            return x + self.fn(self.norm(x), **kwargs)
        b, _, _ = x.shape
        n_keep = max(int(b * self.keep_prob), 1)
        indices = torch.randperm(b, device=x.device)[:n_keep]
        for k, v in kwargs.items():
            if v is not None and k != "attn_mask":
                kwargs[k] = v[indices]
        return torch.index_add(
            x,
            dim=0,
            source=self.fn(self.norm(x[indices]), **kwargs),
            index=indices,
            alpha=b / n_keep,
        )


# copied from CAPI
def rotate_half(x: Float[Tensor, "*b d"]) -> Float[Tensor, "*b d"]:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# ========= Slightly adapted CAPI code below =========


class Rope(nn.Module):
    """
    3D rotary position embedding over (t, y, x)
    """

    def __init__(
        self,
        dim: int,
        max_freq: float | int = 7,
        min_freq: float | int = 7e-4,
    ):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.freqs = nn.Parameter(torch.empty(3, self.dim))

        # added this just in case
        self._device_weight_init()

    def _device_weight_init(self):
        # For a head_dim=64, we split it into 3 and leave the rest unrotated.
        # That is, 20 dims for t, 20 dims for y, 20 dims for x, and 4 dims are zeros.
        # This is compatible with CAPI's ROPE as well as DINO-world (Appendix A.1).
        third = 2 * (self.dim // 6)  # the remainder dims are left as zeros = unrotated
        freqs_1d = self.max_freq * (self.max_freq / self.min_freq) ** torch.linspace(
            0, -1, third // 2, device=self.freqs.device
        )
        freqs_1d = torch.cat([freqs_1d, freqs_1d])
        freqs_3d = torch.zeros(3, self.dim)
        freqs_3d[0, :third] = freqs_1d
        freqs_3d[1, third : 2 * third] = freqs_1d
        freqs_3d[2, 2 * third : 3 * third] = freqs_1d
        self.freqs.data.copy_(freqs_3d * 2 * torch.pi)

    def forward(
        self,
        x: Float[Tensor, "*b d"],
        coords: Float[Tensor, "*b 3"],
    ) -> Float[Tensor, "*b d"]:
        angle = coords @ self.freqs
        return x * angle.cos() + rotate_half(x) * angle.sin()


class Attention(nn.Module):
    """
    Cross-attention with 3D RoPE, extending CAPI's 2D Rope to (time, y, x)
    continuous coordinates.

    CHANGELOG:
    - Uses 3D RoPE instead of 2D.
    - Accepts `attn_mask` to allow for DINO-world's block-triangular mask (frame t only attends to frames <= t).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        context_dim: int | None = None,
        rope_kwargs: Mapping = {},
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        context_dim = context_dim or dim

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = Rope(dim=head_dim, **rope_kwargs)

    def forward(
        self,
        x: Float[Tensor, "b n_q d"],
        coords: Float[Tensor, "b n_q 3"],
        context: Float[Tensor, "b n_k d_k"] | None = None,
        context_coords: Float[Tensor, "b n_k 3"] | None = None,
        attn_mask: Bool[Tensor, "n_q n_k"] | None = None,
    ) -> Float[Tensor, "b n_q d"]:
        if context is None or context_coords is None:
            context = x
            context_coords = coords

        b, n_q, d = x.shape
        b, n_k, _ = context.shape
        h = self.num_heads

        q = self.q_proj(x).reshape(b, n_q, h, d // h).transpose(1, 2)
        k = self.k_proj(context).reshape(b, n_k, h, d // h).transpose(1, 2)
        v = self.v_proj(context).reshape(b, n_k, h, d // h).transpose(1, 2)

        q = self.rope(q, coords[:, None, :, :])
        k = self.rope(k, context_coords[:, None, :, :])

        if attn_mask is not None and attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(b, n_q, d)
        x = self.proj(x)
        return x


class Block(nn.Module):
    """
    Cross-attention block with MLP.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float | int,
        norm_layer: Callable[[int], nn.Module],
        context_dim: int | None,
        drop_path_type: str = "efficient",
        attn_kwargs: Mapping = {},
    ) -> None:
        super().__init__()
        residual_module = {
            "naive": NaiveResidual,
            "efficient": EfficientResidual,
        }[drop_path_type]
        self.residual1 = residual_module(
            drop_path,
            norm_layer(dim),
            Attention(
                dim,
                context_dim=context_dim,
                **attn_kwargs,
            ),
        )
        self.residual2 = residual_module(
            drop_path,
            norm_layer(dim),
            Mlp(in_features=dim),
        )

    def forward(
        self,
        x: Float[Tensor, "b n d"],
        context: Float[Tensor, "b n_k d_k"] | None = None,
        coords: Float[Tensor, "b n 3"] | None = None,
        context_coords: Float[Tensor, "b n_k 3"] | None = None,
        attn_mask: Bool[Tensor, "b n n_k"] | None = None,
    ) -> Float[Tensor, "b n d"]:
        x = self.residual1(
            x,
            context=context,
            coords=coords,
            context_coords=context_coords,
            attn_mask=attn_mask,
        )
        x = self.residual2(x)
        return x


# from https://github.com/facebookresearch/mae/blob/main/models_mae.py
def _init_weights(m: nn.Module, xavier_gain=1) -> None:
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        init.xavier_uniform_(m.weight, gain=xavier_gain)
        if isinstance(m, nn.Linear) and m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm | nn.RMSNorm) and m.elementwise_affine:
        init.constant_(m.weight, 1.0)
        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias, 0)
    if hasattr(m, "_device_weight_init"):
        m._device_weight_init()


class CrossAttentionPredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        encoder_dim: int,
        drop_path_rate: float = 0.0,
        rope_kwargs: dict = {},
    ) -> None:
        """
        :param dim: Embedding dimension of the predictor (decoder).
        :param encoder_dim: Embedding dimension of the encoder features.
        :param depth: Number of cross-attention blocks.
        :param num_heads: Number of attention heads.
        :param drop_path_rate: Drop path rate.
        """
        super().__init__()
        self.dim = dim
        self.encoder_dim = encoder_dim
        self.depth = depth
        self.num_heads = num_heads

        norm_layer = partial(nn.LayerNorm, eps=1e-5)
        self.ctx_proj = nn.Linear(self.encoder_dim, self.dim, bias=True)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.dim,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    context_dim=self.dim,
                    attn_kwargs={"num_heads": num_heads, "rope_kwargs": rope_kwargs},
                )
                for _ in range(depth)
            ],
        )
        self.mask_token = nn.Parameter(torch.empty(1, self.dim))
        self.dec_norm = norm_layer(self.dim)
        self.dec_proj = nn.Linear(self.dim, self.encoder_dim, bias=True)

        # init
        self.apply(_init_weights)
        init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        tokens: Float[Tensor, "B T H W D_enc"],
        coords: Float[Tensor, "B N 3"],
        ctx_mask: Bool[Tensor, "B N_pred N"],
        coords_query: Float[Tensor, "B N_pred 3"],
    ) -> Float[Tensor, "B N_pred D_enc"]:
        """
        :param tokens: Patch tokens for a sequence of frames, in encoder space.
        :param coords: (time, y, x) coordinates for all tokens.
        :param ctx_mask: Context mask over tokens for each prediction.
        :param coords_query: Coordinates for the query tokens.
        """
        B, T, H, W, D_enc = tokens.shape
        _, N_pred, N = ctx_mask.shape
        assert D_enc == self.encoder_dim

        tokens = tokens.flatten(1, 3)
        tokens = self.ctx_proj(tokens)

        # coords_query = coords[pred_mask].view(B, N_pred, 3)
        x = self.mask_token[None].expand(B, N_pred, -1)

        for blk in self.blocks:
            x = blk(
                x,
                context=tokens,
                coords=coords_query,
                context_coords=coords,
                attn_mask=ctx_mask,
            )
        x = self.dec_norm(x)
        x = self.dec_proj(x)
        return x


class DinoWorldModel(nn.Module):
    def __init__(self, encoder: TimmViTEncoder, predictor: CrossAttentionPredictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(
        self,
        frames: Float[Tensor, "b t c H W"],
        timestamps: Float[Tensor, "b t"],
    ) -> dict:
        """
        :param frames: Input video frames.
        :param timestamps: Timestamps in seconds.
        """
        frame_tokens = self.encoder(frames)
        b, _, h, w, d = frame_tokens.shape

        coords = get_all_video_coords(timestamps, h, w)
        pred_mask, ctx_mask = next_frame_prediction_masks_compiled(coords)

        # Slice to get query coords (frames 1..T)
        coords_query = coords[:, 1:].flatten(1, 3)  # (B, N_pred, 3)

        preds = self.predictor(
            tokens=frame_tokens,
            coords=coords.flatten(1, 3),
            ctx_mask=ctx_mask,
            coords_query=coords_query,
        )
        # Slice targets (frames 1..T)
        targets = frame_tokens[:, 1:].flatten(1, 3)  # (B, N_pred, D)

        loss = torch.nn.functional.smooth_l1_loss(preds, targets, beta=0.1)
        return {"loss": loss}
