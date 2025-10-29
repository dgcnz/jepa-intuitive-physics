import torch
import logging
from intphyseval.model.sigma.sigma import PretrainVisionTransformer, FeatureExtractor
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm
from torch import nn
from einops import rearrange
from functools import partial

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class DINOVideoEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "dino_vitb16",
        input_size: int = 224,
        output_dim: int = 768,
        normalize_targets: bool = True,
    ):
        super(DINOVideoEncoder, self).__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.normalize_targets = normalize_targets
        assert normalize_targets, "Currently only support normalize_targets=True"
        assert self.output_dim == 768, "DINO's output dim is 768"

        pretraining = torch.hub.load("facebookresearch/dino:main", model_name)
        feature_extraction_model = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0
        )
        msg = feature_extraction_model.load_state_dict(
            pretraining.state_dict(), strict=True
        )
        print(msg)
        feature_extraction_model = FeatureExtractor(
            feature_extraction_model, input_size, 16
        )
        feature_extraction_model.eval()
        self.model = feature_extraction_model

    def forward(self, x, full_mask):
        permuted_video = x.permute(0, 2, 1, 3, 4)
        bs, nf, _, h, w = permuted_video.shape
        permuted_video = permuted_video[:, ::2].flatten(0, 1)
        features = self.model(permuted_video)
        _, np, dim = features.shape
        features = features.reshape(bs, nf // 2, np, dim)
        features_squeeze = rearrange(features, "b n o c -> b (n o) c")
        dino_targets = (
            features_squeeze - features_squeeze.mean(dim=-2, keepdim=True)
        ) / (features_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        return dino_targets


class SIGMAVideoEncoder(nn.Module):
    """Wrapper to use SIGMA encoder as target encoder (self-distillation)"""

    def __init__(self, sigma_model):
        super(SIGMAVideoEncoder, self).__init__()
        self.sigma_model = sigma_model

    def forward(self, x, full_mask):
        # Process all tokens (pass inverted full_mask = all False)
        enc_out, _ = self.sigma_model.encoder(x, ~full_mask)
        return enc_out


class MLPEncoder(torch.nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        tubelet_size: int = 2,
        output_dim: int = 256,
        normalize_targets: bool = True,
    ):
        super().__init__()
        C = 3
        input_dim = C * (patch_size**2) * tubelet_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, output_dim),
        )
        self.normalize_targets = normalize_targets
        assert self.normalize_targets, "Currently only support normalize_targets=True"
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

    def forward(self, videos, full_mask):
        unnorm_videos = videos * self.std + self.mean  # in [0, 1]
        videos_squeeze = rearrange(
            unnorm_videos,
            "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
            p0=self.tubelet_size,
            p1=self.patch_size,
            p2=self.patch_size,
        )
        videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)) / (
            videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
        )
        videos_patch = rearrange(videos_norm, "b n p c -> b n (p c)")
        return self.head(videos_patch)


class SIGMA(torch.nn.Module):
    def __init__(
        self,
        encoder: PretrainVisionTransformer,
        target_encoder: DINOVideoEncoder,
        num_frames: int,
        img_size: int,
        patch_size: int,
        target_type: str, # dino_v1 or mlp
        loss_func: str = "SWAV",
    ):
        super().__init__()
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size

        self.encoder = encoder
        self.target_encoder = target_encoder
        self.target_type = target_type
        assert target_type in ["dino_v1", "mlp"], "target_type must be 'dino_v1' or 'mlp'" 
        self.loss_func = loss_func
        assert self.loss_func == "SWAV", "Currently only support SWAV loss"

    def forward(self, pieces, masks_enc, masks_pred, full_mask):
        all_features = self.target_encoder(pieces, full_mask)
        B, _, C = all_features.shape
        targets = all_features[masks_pred].reshape(B, -1, C)
        outputs, (scores1, q1), (scores2, q2) = self.encoder(
            pieces, masks_pred, targets
        )
        targets = q1.argmax(dim=-1)
        preds = scores2 / 0.1
        return preds, targets.long()

    def freeze(self):
        self.target_encoder.eval()
        self.encoder.eval()
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

    def load_ckpt(
        self,
        ckpt: str,
    ):
        logger.info(f"Loading SIGMA checkpoint from {ckpt}")
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)

        # Encoder
        self.encoder.load_state_dict(checkpoint["model"], strict=True)

        if self.target_type == "mlp":
            logger.info("Loading MLP target encoder weights")
            mlp_ckpt = {
                k: v for k, v in checkpoint["model"].items() if k.startswith("head")
            }
            self.target_encoder.load_state_dict(mlp_ckpt, strict=True)
            self.encoder.head = torch.nn.Identity()

        del checkpoint
        return self


def sigma_with_mlp(p="/mnt/sdb1/checkpoint/intphys/ssv2_vit_b_sigma_with_mlp.pth"):
    # batch_size=42,
    # model="pretrain_videomae_base_patch16_224",
    # decoder_depth=4,
    # mask_type="tube",
    # target_type="mlp",
    # loss_func="SWAV",
    # mask_ratio=0.9,
    # input_size=224,
    # drop_path=0.0,
    # normlize_target=True,
    # memory_size=0,
    # distillation_teacher="dino_s",
    # num_prototypes=4000,
    # sinkhorn_eps=0.05,
    # sinkhorn_iterations=10,
    # color_jitter=0.0,
    # num_frames=16,
    # sampling_rate=2,
    # world_size=4,
    # window_size=(8, 14, 14),
    # patch_size=(16, 16),
    m = SIGMA(
        encoder=PretrainVisionTransformer(
            img_size=224,
            patch_size=16,
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_num_classes=0,
            decoder_embed_dim=384,
            decoder_num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            # ***
            decoder_depth=4,
            mask_ratio=0.9,
            mask_type="tube",
            target_type="mlp",
            loss_func="SWAV",
            memory_size=0,
            num_prototypes=4000,
            sinkhorn_iterations=10,
            eps=0.05,
            kwindow=1,
            # ***
            decoder_num_classes=256,  # from DINOv1
        ),
        target_encoder=MLPEncoder(
            patch_size=16,
            tubelet_size=2,
            output_dim=256,
        ),
        num_frames=16,
        img_size=224,
        patch_size=16,
        target_type="mlp",
        loss_func="SWAV",
    )

    m.load_ckpt(p)
    return m


def sigma_with_dino(
    p="/mnt/sdb1/checkpoint/intphys/sigma_ssv2_vit_b_sigma_with_dino.pth",
):
    # batch_size=40,
    # model="pretrain_videomae_base_patch16_224",
    # decoder_depth=4,
    # mask_type="tube",
    # target_type="dino_v1",
    # loss_func="SWAV",
    # mask_ratio=0.9,
    # input_size=224,
    # drop_path=0.0,
    # normlize_target=True,
    # chromatic_correction=False,
    # gray_scale_prob=0.0,
    # kwindow=1,
    # memory_size=0,
    # distillation_teacher="dino_b",
    # num_prototypes=3000,
    # sinkhorn_eps=0.03,
    # sinkhorn_iterations=10,
    # color_jitter=0.0,
    # num_frames=16,
    # sampling_rate=2,
    # world_size=12,
    # window_size=(8, 14, 14),
    # patch_size=(16, 16),
    m = SIGMA(
        encoder=PretrainVisionTransformer(
            img_size=224,
            patch_size=16,
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_num_classes=0,
            decoder_embed_dim=384,
            decoder_num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            # ***
            decoder_depth=4,
            mask_ratio=0.9,
            mask_type="tube",
            target_type="dino_v1",
            loss_func="SWAV",
            memory_size=0,
            num_prototypes=3000,
            sinkhorn_iterations=10,
            eps=0.03,
            kwindow=1,
            # ***
            decoder_num_classes=768,  # from DINOv1
        ),
        target_encoder=DINOVideoEncoder("dino_vitb16", input_size=224),
        num_frames=16,
        img_size=224,
        patch_size=16,
        target_type="dino_v1",
        loss_func="SWAV",
    )

    m.load_ckpt(p)
    return m


if __name__ == "__main__":
    sigma_with_dino()
    sigma_with_mlp()
