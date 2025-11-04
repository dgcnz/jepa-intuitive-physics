import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from tqdm import tqdm
from einops import rearrange
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from jaxtyping import Float, Int
from typing import Callable
from torch import Tensor
from intphyseval.utils import (
    get_breaking_points,
    get_matches,
    get_time_masks,
    pad_tensors,
)

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import lightning as L
import wandb
import rootutils


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

OmegaConf.register_new_resolver("eval", eval)

logging.basicConfig()

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def rearrange_clips(
    clips: Float[Tensor, "V C T H W"],
    model_num_frames: int,
    stride: int,
):
    """Window a multi-video clip and infer matching pairs.

    Returns: pieces [B, C, T, H, W], num_videos, matches (list of index pairs)
    """
    num_videos = clips.shape[0]
    if num_videos == 2:
        matches = [[0, 1]]
    else:
        bps = get_breaking_points(clips)
        matches = get_matches(bps)
    pieces = (
        clips.unfold(2, model_num_frames, stride)
        .permute(0, 2, -1, 1, 3, 4)
        .contiguous()
    )
    pieces = pieces.flatten(0, 1)
    pieces = rearrange(pieces, "b t c h w -> b c t h w").contiguous()
    return pieces, matches


def create_masks(
    context_length: int,
    model_num_frames: int,
    patch_size: int,
    B: int,
    as_bool: bool,
):
    """Create encoder/prediction/full masks for given batch tokenization size."""
    m, m_, full_m = get_time_masks(
        context_length,
        spatial_size=(patch_size, patch_size),
        temporal_dim=model_num_frames,
        as_bool=as_bool,
    )
    full_mask = full_m.unsqueeze(0)
    m = m.unsqueeze(0)
    m_ = m_.unsqueeze(0)

    masks_enc = m.repeat(B, 1)
    masks_pred = m_.repeat(B, 1)
    full_mask = full_mask.repeat(B, 1)
    return masks_enc, masks_pred, full_mask


def cross_entropy_sk(preds, targets):
    return F.cross_entropy(preds.permute(0, 3, 1, 2), targets, reduction="none").mean(2)


def cross_entropy_sk_dense(
    preds: Float[Tensor, "V num_windows N num_prototypes"], targets
) -> Float[Tensor, "V num_windows N"]:
    return F.cross_entropy(preds.permute(0, 3, 1, 2), targets, reduction="none")


def l1_features(preds, targets):
    return F.l1_loss(preds, targets, reduction="none").mean((2, 3))


def l1_dense_features(
    preds: Float[Tensor, "V num_windows N D"], targets
) -> Float[Tensor, "V num_windows N"]:
    return F.l1_loss(preds, targets, reduction="none").mean(-1)


@torch.no_grad()
def extract_losses_single(
    fabric: L.Fabric,
    net,
    loader: DataLoader,
    context_length: int,
    stride: int,
    patch_size: int,
    model_num_frames: int,
    mask_as_bool: bool,
    surprise: Callable[[tuple[Tensor, Tensor]], Tensor],
):
    all_labels = []
    all_losses = []

    for batch in tqdm(loader, disable=not fabric.is_global_zero, mininterval=10):
        assert batch[0].shape[0] == 1, "Batch size > 1 not supported"
        clips: Float[Tensor, "V C T H W"] = batch[0][0]
        labels: Int[Tensor, "V"] = batch[1][0]
        V = clips.shape[0]  # num videos (4 for intphys, 2 elsewhere)

        pieces, matches = rearrange_clips(
            clips=clips,
            model_num_frames=model_num_frames,
            stride=stride,
        )
        masks_enc, masks_pred, full_mask = create_masks(
            context_length=context_length,
            model_num_frames=model_num_frames,
            patch_size=patch_size,
            B=pieces.shape[0],
            as_bool=mask_as_bool,
        )
        masks_enc, masks_pred, full_mask = fabric.to_device(
            (masks_enc, masks_pred, full_mask)
        )

        preds, targets = net(pieces, masks_enc, masks_pred, full_mask)
        # Reshape [B, N, D] -> [V, num_windows, N, D]
        num_windows = preds.shape[0] // V
        preds = preds.unflatten(0, (V, num_windows))
        targets = targets.unflatten(0, (V, num_windows))
        loss = surprise(preds, targets).detach()

        losses = loss.unsqueeze(1)  # [num_videos, n_ctxt=1, n_windows]

        for match in matches:
            all_losses.append(losses[match])
            all_labels.append(labels[match])

    return all_losses, all_labels


def compute_metrics(losses, labels):
    metrics = {}
    loss_real = losses[torch.where(labels == 1)]
    loss_fake = losses[torch.where(labels == 0)]

    acc_pairwise_mean = (
        (loss_real.mean(1) < loss_fake.mean(1)).sum() / loss_real.shape[0] * 100
    ).item()
    acc_pairwise_max = (
        (loss_real.max(1)[0] < loss_fake.max(1)[0]).sum() / loss_real.shape[0] * 100
    ).item()
    metrics["Relative Accuracy (avg)"] = acc_pairwise_mean
    metrics["Relative Accuracy (max)"] = acc_pairwise_max

    data1 = loss_real.max(1)[0]
    data2 = loss_fake.max(1)[0]
    thresh = data1.sort()[0][int(np.ceil(0.90 * len(data1)))]
    accuracy_abs = (
        ((data1 < thresh).sum() + (data2 > thresh).sum())
        / (data1.shape[0] + data2.shape[0])
        * 100
    ).item()
    metrics["Absolute Accuracy (max)"] = accuracy_abs
    metrics["Classifier threhshold"] = thresh

    precision_max, recall_max, _ = precision_recall_curve(labels, -losses.max(1)[0])
    precision_mean, recall_mean, _ = precision_recall_curve(labels, -losses.mean(1))
    metrics["AUPRC (avg)"] = auc(recall_mean, precision_mean)
    metrics["AUPRC (max)"] = auc(recall_max, precision_max)

    fpr_max, tpr_max, _ = roc_curve(labels, -losses.max(1)[0])
    fpr_mean, tpr_mean, _ = roc_curve(labels, -losses.mean(1))
    metrics["AUROC (avg)"] = auc(fpr_mean, tpr_mean)
    metrics["AUROC (max)"] = auc(fpr_max, tpr_max)
    return metrics


def sync_outputs(fabric: L.Fabric, all_losses, all_labels):
    lengths = [l.size(-1) for l in all_losses]  # noqa
    max_length = torch.tensor([max(lengths)], device=fabric.device)
    max_length = fabric.all_reduce(max_length, reduce_op="max")

    all_losses = torch.concat(pad_tensors(all_losses, max_length.item()))
    all_labels = torch.concat(all_labels)

    # Gather + metrics
    gathered_losses = fabric.all_gather(all_losses)
    gathered_labels = fabric.all_gather(all_labels)
    # Flatten world dimension if present (world, N, ...)->(world*N,...)
    if gathered_losses.dim() >= 2 and gathered_losses.size(0) == fabric.world_size:
        gathered_losses = gathered_losses.flatten(0, 1)
    if gathered_labels.dim() >= 2 and gathered_labels.size(0) == fabric.world_size:
        gathered_labels = gathered_labels.flatten(0, 1)
    all_losses = gathered_losses.cpu()
    all_labels = gathered_labels.cpu()
    return all_losses, all_labels


def setup(cfg):
    fabric = L.Fabric(**instantiate(cfg.trainer))

    fabric._loggers[0].log_hyperparams(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    # print(OmegaConf.to_yaml(cfg, resolve=True))

    L.seed_everything(cfg.seed, workers=True)

    # TODO: This is getting pickling errors with SLURM launcher,
    # so disabling for now.

    # # Set float32 matmul precision
    # if cfg.get("float32_matmul_precision"):
    #     torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    # if cfg.get("allow_tf32", False):
    #     torch.backends.cuda.matmul.allow_tf32 = True

    # # cuDNN optimization
    # if cfg.get("cudnn_benchmark", False):
    #     import torch.backends.cudnn as cudnn
    #     cudnn.benchmark = True

    fabric.launch()
    return fabric


def run_eval(cfg):
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fabric = setup(cfg)
    log.info(f"[{fabric.global_rank}/{fabric.world_size}] Initialized")
    log.info(cfg.tags)
    log.info(f"{cfg.data.name}:{cfg.data.property}")

    with fabric.init_module(empty_init=True):
        net = instantiate(cfg.model.net)
    dataloader = instantiate(cfg.data.dataloader)

    if cfg.get("compile", False):
        net = torch.compile(net, fullgraph=True, dynamic=False)

    loader = fabric.setup_dataloaders(dataloader)
    net = fabric.setup(net)

    if cfg.model.pretrained:
        net.load_ckpt(**cfg.model.ckpt_kwargs)
    net.freeze()

    if fabric.is_global_zero:
        log.info("Running eval")

    # Single run over provided loader
    surprises = {
        "cross_entropy_sk": cross_entropy_sk,
        "cross_entropy_sk_dense": cross_entropy_sk_dense,
        "l1": l1_features,
        "l1_dense": l1_dense_features,
    }
    surprise = surprises[cfg.surprise]
    all_losses, all_labels = extract_losses_single(
        fabric=fabric,
        net=net,
        loader=loader,
        context_length=cfg.context_length,
        stride=cfg.stride,
        patch_size=net.patch_size,
        model_num_frames=net.num_frames,
        mask_as_bool=cfg.mask_as_bool,
        surprise=surprise,
    )
    if fabric.is_global_zero:
        log.info("Syncing outputs")

    all_losses, all_labels = sync_outputs(fabric, all_losses, all_labels)

    if fabric.is_global_zero:
        log.info("Saving and computing metrics")

    torch.save(
        {
            "block": cfg.data.property,
            "frame_step": cfg.data.frame_step,
            "context_length": cfg.context_length,
            "losses": all_losses,
            "labels": all_labels,
        },
        output_dir / "losses.pth",
    )
    if not cfg.compute_metrics:
        return
    metrics = compute_metrics(all_losses[:, 0], all_labels)
    if fabric.is_global_zero:
        fabric.log_dict(metrics)


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    try:
        run_eval(cfg)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.exception(e)
        raise e
    finally:
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()
