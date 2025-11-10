import torch
import logging
from torch import nn
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from intphyseval.model.mae.videomae import PretrainVisionTransformer
from functools import partial

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class PixelTargetEncoder(nn.Module):
    """
    Target encoder for pixel-based reconstruction.
    Computes normalized pixel patches as targets following VideoMAE approach.
    """

    def __init__(
        self,
        patch_size: int = 16,
        tubelet_size: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        # ImageNet normalization constants
        self.register_buffer(
            "mean",
            torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1, 1),
            persistent=False,
        )

    def forward(self, pieces, full_mask):
        """
        Args:
            pieces: Video tensor [B, C, T, H, W] (normalized with ImageNet stats)
            full_mask: Not used, for interface compatibility

        Returns:
            Normalized pixel patches [B, N_patches, patch_dim]
            where patch_dim = tubelet_size * patch_size * patch_size * 3
        """
        # Unnormalize from ImageNet stats to [0, 1]
        unnorm_videos = pieces * self.std + self.mean

        # Patchify: convert to [B, num_patches, pixels_per_patch, channels]
        videos_squeeze = rearrange(
            unnorm_videos,
            "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
            p0=self.tubelet_size,
            p1=self.patch_size,
            p2=self.patch_size,
        )

        # Per-patch normalization (mean=0, std=1)
        var = (
            videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
        )
        mean = videos_squeeze.mean(dim=-2, keepdim=True)
        videos_norm = (videos_squeeze - mean) / var

        # Flatten patches: [B, num_patches, patch_dim]
        videos_patch = rearrange(videos_norm, "b n p c -> b n (p c)")

        return videos_patch


class LatentTargetEncoder(nn.Module):
    """
    Target encoder for latent feature prediction.
    Uses VideoMAE encoder/decoder features as targets.
    """

    def __init__(
        self,
        encoder: PretrainVisionTransformer,
        decoder_blocks: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder_blocks = decoder_blocks

    def forward(self, pieces, full_mask):
        """
        Args:
            pieces: Video tensor [B, C, T, H, W]
            full_mask: Boolean mask [B, N_patches] where True = visible

        Returns:
            Latent features [B, N_patches, feature_dim]
        """
        # Run encoder with inverted mask (VideoMAE convention: True = masked)
        # full_mask has True for visible, so we invert it
        latent_features = self.encoder(
            pieces,
            ~full_mask,  # Invert: True = masked
            decoder_blocks=self.decoder_blocks,
        )
        return latent_features


class VideoMAE(nn.Module):
    """
    VideoMAE wrapper following SIGMA/JEPA pattern.
    Supports both pixel reconstruction and latent feature prediction.
    """

    def __init__(
        self,
        encoder: PretrainVisionTransformer,
        target_encoder: nn.Module,  # PixelTargetEncoder or LatentTargetEncoder
        num_frames: int,
        img_size: int,
        patch_size: int,
        tubelet_size: int = 2,
        decoder_blocks: int = -1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.decoder_blocks = decoder_blocks

        self.encoder = encoder
        self.target_encoder = target_encoder

    def forward(self, pieces, masks_enc, masks_pred, full_mask):
        """
        Unified forward interface for VideoMAE.

        Args:
            pieces: Video tensor [B, C, T, H, W]
            masks_enc: Boolean mask for encoder (not used for MAE, uses masks_pred)
            masks_pred: Boolean mask [B, N_patches] where True = predict this patch
            full_mask: Boolean mask [B, N_patches] where True = visible (all True for targets)

        Returns:
            preds: Predictions for masked patches [B, N_masked, feature_dim]
            targets: Target features for masked patches [B, N_masked, feature_dim]
        """
        # Compute targets from all patches
        all_features = self.target_encoder(pieces, full_mask)
        B, _, C = all_features.shape

        # Extract targets for predicted patches
        targets = all_features[masks_pred].reshape(B, -1, C)

        # Compute predictions (encoder processes only visible patches)
        # VideoMAE convention: True = masked, so use masks_pred directly
        preds = self.encoder(pieces, masks_pred, decoder_blocks=self.decoder_blocks)

        return preds, targets

    def freeze(self):
        """Freeze all parameters for evaluation."""
        self.encoder.eval()
        self.target_encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def load_ckpt(
        self,
        ckpt: str,
        enc_checkpoint_key: str = "model",
    ):
        """
        Load checkpoint into encoder.

        Args:
            ckpt: Path to checkpoint file
            enc_checkpoint_key: Key in checkpoint dict containing model weights
        """
        logger.info(f"Loading VideoMAE checkpoint from {ckpt}")
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)

        # Load encoder weights
        if enc_checkpoint_key in checkpoint:
            state_dict = checkpoint[enc_checkpoint_key]
        else:
            state_dict = checkpoint

        self.encoder.load_state_dict(state_dict, strict=True)
        logger.info("VideoMAE checkpoint loaded successfully")

        del checkpoint
        return self


def videomae_v2_giant(
    ckpt_path: str = "/mnt/sdb1/checkpoint/intphys/vit_g_hybrid_pt_1200e.pth",
    num_frames: int = 16,
):
    """
    Helper function to create VideoMAE v2 Giant model.

    Note: The pretrained checkpoint has decoder_depth=4, not 8.
    """
    m = VideoMAE(
        encoder=PretrainVisionTransformer(
            img_size=224,
            patch_size=14,
            encoder_in_chans=3,
            encoder_num_classes=0,
            encoder_embed_dim=1408,
            encoder_depth=40,
            encoder_num_heads=16,
            decoder_embed_dim=512,
            decoder_depth=4,  # Pretrained checkpoint has 4, not 8
            decoder_num_heads=8,
            decoder_num_classes=1176,
            mlp_ratio=48/11,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            init_values=0.0,
            use_learnable_pos_emb=False,
            use_checkpoint=False,
            tubelet_size=2,
        ),
        target_encoder=PixelTargetEncoder(
            patch_size=14,
            tubelet_size=2,
        ),
        num_frames=num_frames,
        img_size=224,
        patch_size=14,
        tubelet_size=2,
        decoder_blocks=-1,
    )

    m.load_ckpt(ckpt_path, enc_checkpoint_key="model")
    return m


if __name__ == "__main__":
    # Test loading
    model = videomae_v2_giant()
    print("VideoMAE v2 Giant loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
