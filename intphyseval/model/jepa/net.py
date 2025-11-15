import torch
import torch.nn.functional as F
from src.masks.utils import apply_masks
import logging
from itertools import product

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class JEPA(torch.nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        target_encoder,
        num_frames: int,
        img_size: int,
        patch_size: int,
        tubelet_size: int,
        normalize_targets=True,
        normalize_enc=False,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = target_encoder
        self.normalize_targets = normalize_targets
        self.normalize_enc = normalize_enc

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
        h = self.target_encoder(pieces, [full_mask])
        if self.normalize_targets:
            h = F.layer_norm(h, (h.size(-1),))  # [B, N, D]
        targets = apply_masks(h, [masks_pred], concat=True)  # [B, N_tgt, D]

        # Context tokens directly from encoder with masking inside forward
        context = self.encoder(pieces, [masks_enc])  # [B, N_ctx, D]
        if self.normalize_enc:
            context = F.layer_norm(context, (context.size(-1),))

        # Predictor outputs target-token predictions
        preds = self.predictor(
            context, targets, [masks_enc], [masks_pred]
        )  # [B, N_tgt, D]

        return preds, targets

    def freeze(self):
        self.target_encoder.eval()
        self.predictor.eval()
        self.encoder.eval()
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.predictor.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

    def load_ckpt(
        self,
        ckpt: str,
        enc_checkpoint_key: str = "encoder",
        target_enc_checkpoint_key: str = "target_encoder",
        pred_checkpoint_key: str = "predictor",
    ):
        def cleanup(sd: dict):
            # torch.compile
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            # for MultiMask checkpoints
            sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
            return sd

        # JEPA's pos_embed is mismatched if num_frames != 16,
        # but for non-ROPE variants it's fixed sin-cos,
        # so we can just skip loading it
        # we'll just replace them with the model's own pos_embed
        def preprocess(sd: dict, module: str):
            assert module in ("encoder", "predictor")
            sd = cleanup(sd)
            pfx = "" if module == "encoder" else "predictor_"
            mod = self.encoder if module == "encoder" else self.predictor
            mod = mod.__getattr__
            if f"{pfx}pos_embed" in sd:  # sin-cos
                sd[f"{pfx}pos_embed"] = mod(f"{pfx}pos_embed")
            else:  # rope
                # v-jepa checkpoints don't use RotaryEmbedding, so we must skip
                # the freqs `blocks.0.attn.h_rotary_emb.freqs`
                for b, d in product(range(len(mod(f"{pfx}blocks"))), ["d", "h", "w"]):
                    key = f"{pfx}blocks.{b}.attn.{d}_rotary_emb.freqs"
                    if key not in sd:
                        sd[key] = mod(f"{pfx}blocks")[b].attn.h_rotary_emb.freqs

            return sd

        logger.info(f"Loading JEPA checkpoint from {ckpt}")
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)

        # Encoder
        enc_state = preprocess(checkpoint[enc_checkpoint_key], "encoder")
        self.encoder.load_state_dict(enc_state, strict=True)

        # Target encoder
        tgt_state = preprocess(checkpoint[target_enc_checkpoint_key], "encoder")
        self.target_encoder.load_state_dict(tgt_state, strict=True)

        # Predictor
        pred_state = preprocess(checkpoint[pred_checkpoint_key], "predictor")
        self.predictor.load_state_dict(pred_state, strict=True)

        del checkpoint
        return self
