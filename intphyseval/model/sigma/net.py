import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from evals.intuitive_physics.videomae import (
    Block,
    PatchEmbed,
    get_sinusoid_encoding_table,
    _cfg,
)
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
import torch.distributed as dist
from einops import rearrange
import timm


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    "pretrain_sigma_small_patch16_224",
    "pretrain_sigma_base_patch16_224",
    "pretrain_sigma_large_patch16_224",
    "pretrain_sigma_huge_patch16_224",
]


class DINOVideoEncoder(nn.Module):
    def __init__(self, model_name: str = "dino_vitb16", input_size: int = 224):
        super(DINOVideoEncoder, self).__init__()
        self.model_name = model_name

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


class PretrainVisionTransformerEncoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        tubelet_size=2,
        use_checkpoint=False,
        mask_type="tube",
        use_learnable_pos_emb=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint
        self.mask_type = mask_type

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x, mask):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)
        mask_pos = None

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        if self.mask_type == "tube":
            x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible
        else:
            # x_vis = utils.mask_fix(x, ~mask)
            x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible

        if self.use_checkpoint:
            for blk in self.blocks:
                x_vis = checkpoint.checkpoint(blk, x_vis)
        else:
            for blk in self.blocks:
                x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis, mask_pos

    def forward(self, x, mask):
        x, mask_pos = self.forward_features(x, mask)
        x = self.head(x)
        return x, mask_pos


class PretrainVisionTransformerDecoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        patch_size=16,
        num_classes=768,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        num_patches=196,
        tubelet_size=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        # assert num_classes == 3 * tubelet_size * patch_size ** 2
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x, return_token_num):
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        if return_token_num > 0:
            x = self.head(
                self.norm(x[:, -return_token_num:])
            )  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x


class FeatureExtractor(torch.nn.Module):
    def __init__(self, vit_model, input_size, patch_size):
        super(FeatureExtractor, self).__init__()
        self.vit_model = vit_model
        self.input_size = input_size
        self.patch_size = patch_size
        self.spatial_resolution = input_size // patch_size
        assert self.spatial_resolution * patch_size == input_size

    def forward(self, x):
        if self.patch_size == 14:
            features = self.vit_model.forward_features(x)[:, 5:]
            bs, np, dim = features.shape
            features = features.reshape(
                bs, self.spatial_resolution, self.spatial_resolution, dim
            ).permute(0, 3, 1, 2)
            features = F.interpolate(features, size=(14, 14), mode="bilinear")
            features = features.flatten(2, -1).permute(0, 2, 1)
        else:
            features = self.vit_model.forward_features(x)[:, 1:]
        return features


class PretrainVisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        encoder_in_chans=3,
        encoder_num_classes=0,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_num_classes=256,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.0,
        use_learnable_pos_emb=False,
        use_checkpoint=False,
        tubelet_size=2,
        n_parts=20,
        mask_ratio=0.5,
        target_type="pixel",
        mask_type="tube",
        memory_size=0,
        loss_func="SWAV",
        num_prototypes=3000,
        world_size=0,
        sinkhorn_iterations=10,
        eps=0.05,
        kwindow=1,
        num_classes=0,  # avoid the error from create_fn in timm
        in_chans=0,  # avoid the error from create_fn in timm
        pretrained_cfg=None,  # avoid the error from create_fn in timm
        pretrained_cfg_overlay=None,  # avoid the error from create_fn in timm
    ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            mask_type=mask_type,
            use_learnable_pos_emb=use_learnable_pos_emb,
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
        )

        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False
        )
        self.world_size = world_size
        self.eps = eps
        self.sinkhorn_iterations = sinkhorn_iterations
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=0.02)

        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim
        )

        self.mask_ratio = mask_ratio
        self.target_type = target_type
        self.mask_type = mask_type
        self.decoder_num_classes = decoder_num_classes

        self.memory_size = memory_size
        if memory_size > 0:
            self.memory = torch.nn.Parameter(
                torch.randn(memory_size, decoder_num_classes)
            )
            self.memory_pred_head = torch.nn.Sequential(
                torch.nn.Linear(decoder_num_classes, decoder_num_classes),
                torch.nn.GELU(),
                torch.nn.Linear(decoder_num_classes, decoder_num_classes),
                torch.nn.GELU(),
                torch.nn.Linear(decoder_num_classes, memory_size),
            )

        def init_proj_mlp(input_dim):
            # return nn.Linear(input_dim, decoder_num_classes)

            return torch.nn.Sequential(
                torch.nn.Linear(input_dim, 1024),
                torch.nn.GELU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.GELU(),
                torch.nn.Linear(1024, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, decoder_num_classes),
            )

        self.loss_func = loss_func
        if loss_func == "SWAV":
            self.num_prototypes = num_prototypes
            self.prototypes = torch.nn.Parameter(
                torch.randn(num_prototypes, decoder_num_classes)
            )

        if ("dino" in target_type) and self.loss_func == "SWAV":
            # self.head = init_proj_mlp(384)
            self.head = nn.Identity()
        elif target_type == "mlp":
            self.head = init_proj_mlp(1536)

    def extract_assignments(self, projected_features, detach=False):
        bs, np, dim = projected_features.shape
        projected_dim = projected_features.shape[-1]
        projected_features = projected_features.reshape(-1, projected_dim)
        normalized_projected_features = F.normalize(projected_features, dim=-1, p=2)

        prototypes = self.prototypes.detach() if detach else self.prototypes

        batch_scores = torch.einsum(
            "bd,nd->bn", normalized_projected_features, prototypes
        )
        batch_q = find_optimal_assignment(
            batch_scores, self.eps, self.sinkhorn_iterations, world_size=self.world_size
        )
        batch_q = batch_q.reshape(bs, np, self.num_prototypes)
        batch_scores = batch_scores.reshape(bs, np, self.num_prototypes)
        return batch_scores, batch_q

    def normalize_prototypes(self):
        with torch.no_grad():
            w = self.prototypes.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.copy_(w)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "mask_token"}

    def forward(self, x, mask, targets):
        B, _, T, _, _ = x.shape
        enc_out, mask_pos = self.encoder(x, mask)  # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(enc_out)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = (
            self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        )

        if self.mask_type == "tube" or self.mask_type == "tube_fgbg":
            pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        elif self.mask_type == "parts":
            pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)

        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)

        return_token = pos_emd_mask.shape[1]

        x_full = torch.cat(
            [x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1
        )  # [B, N, C_d]
        x = self.decoder(
            x_full, return_token
        )  # [B, N_mask, 3 * 16 * 16] #todo change last argument

        # DPC memory implementation for DPC
        if self.memory_size > 0:
            mem_query = self.memory_pred_head(x)
            softmax = F.softmax(mem_query, dim=-1)
            x = torch.einsum("bnm,lmd->bnld", softmax, self.memory.unsqueeze(0))
            x = x.squeeze(2)

        if "mlp" in self.target_type or (
            ("dino" in self.target_type) and self.loss_func == "SWAV"
        ):
            proj = self.head(targets)

        if "mha" in self.target_type:
            proj = proj.reshape(B, T // 2, 196, self.decoder_num_classes).flatten(0, 1)
            proj = self.mha(
                self.key(proj),
                self.value(proj),
                self.query(proj),
                attn_mask=self.attn_mask.to(proj.device),
            )[0]
            proj = proj.reshape(B, T // 2, 196, self.decoder_num_classes).flatten(1, 2)
            proj = self.head2(proj)
            proj = proj[mask].reshape(B, -1, self.decoder_num_classes)
            # attention mask

        if self.loss_func == "SWAV":
            self.normalize_prototypes()
            scores1, q1 = self.extract_assignments(proj, detach=False)
            scores2, q2 = self.extract_assignments(x, detach=True)

            return x, (scores1, q1), (scores2, q2)
        else:
            return x, (None, None), (None, None)


@register_model
def pretrain_sigma_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_sigma_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
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
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_sigma_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_sigma_huge_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_embed_dim=640,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@torch.no_grad()
def sinkhorn(Q: torch.Tensor, nmb_iters: int, world_size=1) -> torch.Tensor:
    with torch.no_grad():
        Q = Q.detach().clone()
        sum_Q = torch.sum(Q)
        if world_size > 1:
            dist.all_reduce(sum_Q)
        Q /= sum_Q
        K, B = Q.shape
        u = torch.zeros(K).to(Q.device)
        r = torch.ones(K).to(Q.device) / K
        c = torch.ones(B).to(Q.device) / (B * world_size)

        if world_size > 1:
            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

        for _ in range(nmb_iters):
            if world_size > 1:
                u = curr_sum
            else:
                u = torch.sum(Q, dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            if world_size > 1:
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def find_optimal_assignment(scores, epsilon, sinkhorn_iterations, world_size=1):
    """
    Computes the Sinkhorn matrix Q.
    :param scores: similarity matrix
    :return: Sinkhorn matrix Q
    """
    with torch.no_grad():
        q = torch.exp(scores / epsilon).t()
        q = sinkhorn(q, sinkhorn_iterations, world_size=world_size)
        # q = torch.softmax(scores / epsilon, dim=0)
        # q = q / q.sum(dim=1, keepdim=True)
    return q
