import torch
import logging
from intphyseval.model.sigma.net import PretrainVisionTransformer, DINOVideoEncoder

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)
"""
        # we're going to do a little hack because I want to touch this codebase the least as possible
        # we're not going to construct the model and the target encoder
        # with args from the function but rather directly from the checkpoint
        args = torch.load(pretrained, weights_only=False, map_location="cpu")["args"]
        assert "dino_v1" == args.target_type, args.target_type
        assert "dino_b" == args.distillation_teacher, args.distillation_teacher
        assert args.normlize_target, "args.normlize_target should be True"
        dec_dim = 768
        # "pretrain_videomae_base_patch16_224"
        # "pretrain_sigma_base_patch16_224"
        # replace _videomae_ by _sigma_, originally as videomae
        # but changed to sigma to avoid namespace conflicts
        assert model_name.replace("sigma", "videomae") == args.model
        encoder = SIGMA.__dict__[model_name](
            drop_path_rate=args.drop_path,
            decoder_depth=args.decoder_depth,
            use_checkpoint=args.use_checkpoint,
            decoder_num_classes=dec_dim,
            target_type=args.target_type,
            mask_ratio=args.mask_ratio,
            mask_type=args.mask_type,
            loss_func=args.loss_func,
            memory_size=args.memory_size,
            num_prototypes=args.num_prototypes,
            world_size=1,  # this is just for the sinkhorn computation, not needed
            sinkhorn_iterations=args.sinkhorn_iterations,
            eps=args.sinkhorn_eps,
            # Some SIGMA checkpoints may not define kwindow; default to 1
            kwindow=getattr(args, "kwindow", 1),
        )
        if sigma_target_enc == "dino":
            target_encoder = SIGMA.DINOVideoEncoder(
                "dino_vitb16", input_size=args.input_size
            )
            target_encoder = target_encoder.to(device)
        elif sigma_target_enc == "sigma":
            target_encoder = None  # Will be created after loading weights
        predictor = None
        assert enc_checkpoint_key == "model"
"""

"""
                    all_features = target_encoder(pieces, full_mask)
                    B, _, C = all_features.shape
                    targets = all_features[masks_pred].reshape(B, -1, C)
                    outputs, (scores1, q1), (scores2, q2) = encoder(
                        pieces, masks_pred, targets
                    )

                    if sigma_loss_type == "cross_entropy":
                        # maybe easier CE, doesn't require sinkhorn-knopp, which is batch-sensitive
                        # this eval uses B=1 per video, so SK might not be ideal
                        scores1 = (scores1 / 0.1).softmax(-1)
                        scores2 = (scores2 / 0.1).softmax(-1)
                        p_v = scores2.view(num_videos, -1, *scores2.shape[1:])
                        p_phi = scores1.view(num_videos, -1, *scores1.shape[1:])
                        loss = (
                            -(p_phi * (p_v.clamp_min(1e-6)).log()).sum(-1).mean(-1)
                        ).detach()
                    elif sigma_loss_type == "cross_entropy_sk":
                        # cross-entropy: predict target assignments (q1) from decoder scores (scores2)
                        q1_hard = q1.argmax(dim=-1)
                        scores2_scaled = scores2 / 0.1

                        scores2_vid = scores2_scaled.view(num_videos, -1, *scores2_scaled.shape[1:])
                        q1_hard_vid = q1_hard.view(num_videos, -1, *q1_hard.shape[1:])

                        loss = F.cross_entropy(
                            scores2_vid.permute(0, 3, 1, 2), q1_hard_vid.long(), reduction='none'
                        ).mean(2).detach()
                    elif sigma_loss_type == "l1_features":
                        # l1 loss like v-jepa, just in case
                        outputs_reshaped = outputs.view(num_videos, -1, *outputs.shape[1:])
                        targets_reshaped = targets.view(num_videos, -1, *targets.shape[1:])
                        loss = (
                            F.l1_loss(outputs_reshaped, targets_reshaped, reduction="none")
                            .mean((2, 3))
                            .detach()
                        )
"""


class SIGMA(torch.nn.Module):
    def __init__(
        self,
        encoder: PretrainVisionTransformer,
        target_encoder: DINOVideoEncoder,
        num_frames: int,
        img_size: int,
        patch_size: int,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size

        self.encoder = encoder
        self.target_encoder = target_encoder

    def forward(self, pieces, masks_enc, masks_pred, full_mask):
        """
        Forward pass without MultiMask wrappers.

        - pieces: [B, C, T, H, W]
        - masks_enc/masks_pred/full_mask: [B, K] index tensors over tokens
        - returns preds/targets shaped [B, N_tokens, D] with B = num_videos * num_windows
        """
        # Target features on full tokens, then select target tokens
        # Note: VisionTransformer accepts a tensor or list of masks. We pass lists to
        # keep apply_masks behavior (concatenation on batch) consistent.

        all_features = self.target_encoder(pieces, full_mask)
        B, _, C = all_features.shape
        targets = all_features[masks_pred].reshape(B, -1, C)
        outputs, (scores1, q1), (scores2, q2) = self.encoder(
            pieces, masks_pred, targets
        )
        targets = q1.argmax(dim=-1)
        preds = scores2 / 0.1
        return preds, targets

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
        print(self.encoder.head)
        print([k for k in checkpoint["model"].keys() if  k.startswith('head')])
        self.encoder.load_state_dict(checkpoint["model"], strict=True)

        # TODO: force DINO target checkpoint be loaded here
        del checkpoint
        return self


if __name__ == "__main__":
    from functools import partial

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
            decoder_num_classes=768, # from DINOv1
        ),
        target_encoder=DINOVideoEncoder("dino_vitb16", input_size=224),
        num_frames=16,
        img_size=224,
        patch_size=16,
    )

    p = "/mnt/sdb1/checkpoint/intphys/sigma_ssv2_vit_b_sigma_with_dino.pth"
    m.load_ckpt(p)


"""
Namespace(batch_size=40, epochs=1601, save_ckpt_freq=20, model='pretrain_videomae_base_patch16_224', decoder_depth=4, mask_type='tube', target_type='dino_v1', loss_func='SWAV', augmentation='multi_scale_crop', mask_ratio=0.9, input_size=224, drop_path=0.0, normlize_target=True, chromatic_correction=False, gray_scale_prob=0.0, kwindow=1, memory_size=0, distillation_teacher='dino_b', num_prototypes=3000, sinkhorn_eps=0.03, sinkhorn_iterations=10, opt='adamw', opt_eps=1e-08, opt_betas=[0.9, 0.95], clip_grad=None, momentum=0.9, weight_decay=0.05, weight_decay_end=0.05, lr=0.00028125, warmup_lr=1.8749999999999998e-06, min_lr=1.8750000000000002e-05, warmup_epochs=40, warmup_steps=-1, use_checkpoint=False, color_jitter=0.0, train_interpolation='bicubic', data_path='/scratch-shared/ssalehi/20bn-something-something-v2/something-something-v2-videos_avi/', mlp_preloading='', data_path_csv='/scratch-shared/ssalehi/20bn-something-something-v2/something-something-v2-annotations/train.csv', imagenet_default_mean_and_std=True, num_frames=16, sampling_rate=2, output_dir='/scratch-shared/ssalehi/snellius_runs/SN_FULL_SSV2_DINO_SWAV_3K_03EPS/', run_name='SN_FULL_SSV2_DINO_SWAV_3K_02EPS', device='cuda', seed=0, resume='/scratch-shared/ssalehi/snellius_runs/SN_FULL_SSV2_DINO_SWAV_3K_03EPS/checkpoint-739.pth', auto_resume=True, start_epoch=740, num_workers=10, pin_mem=True, world_size=12, local_rank=-1, dist_on_itp=False, dist_url='env://', rank=0, gpu=0, distributed=True, dist_backend='nccl', window_size=(8, 14, 14), patch_size=(16, 16))
"""
