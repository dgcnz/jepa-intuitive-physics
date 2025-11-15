import logging
from pathlib import Path
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import random
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, roc_curve, auc

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import lightning as L
import rootutils
from jaxtyping import Float
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from intphyseval.data.sliding_window_dataset import SlidingWindowVideoPredictionDataset
from intphyseval.utils import get_time_masks

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
OmegaConf.register_new_resolver("eval", eval)

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def l1_features(
    preds: Float[Tensor, "B N D"], targets: Float[Tensor, "B N D"]
) -> Float[Tensor, "B"]:  # noqa: F821
    return F.l1_loss(preds, targets, reduction="none").mean((-2, -1))


def l1_features_dense(preds, targets):
    return F.l1_loss(preds, targets, reduction="none").mean((-1))


def cross_entropy_sk_dense(
    preds: Float[Tensor, "B N C"], targets: Float[Tensor, "B N"]
) -> Float[Tensor, "B"]:  # noqa: F821
    return F.cross_entropy(preds.permute(0, 2, 1), targets, reduction="none")


def cross_entropy_sk(
    preds: Float[Tensor, "B N C"], targets: Float[Tensor, "B N"]
) -> Float[Tensor, "B"]:  # noqa: F821
    return F.cross_entropy(preds.permute(0, 2, 1), targets, reduction="none").mean(-1)


@torch.no_grad()
def compute_losses(
    net: torch.nn.Module,
    loader: DataLoader,
    surprise_fn,
) -> Tensor:
    all_losses = []
    for samples, masks_enc, masks_pred, full_mask in tqdm(loader, mininterval=10):
        # samples: [B, C, T, H, W]
        preds, targets = net(samples, masks_enc, masks_pred, full_mask)
        loss = surprise_fn(preds, targets).detach()  # [B, ...]
        all_losses.append(loss.cpu())
    return torch.cat(all_losses, dim=0)


def compute_metrics(losses: Float[Tensor, "B S"], labels: Float[Tensor, " B"]) -> dict:
    """
    Compute pairwise comparison metrics.

    :param losses: Tensor of losses
    :param labels: Tensor of labels
    :return: Dictionary of metrics
    """
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
    metrics["Classifier threshold"] = thresh

    precision_max, recall_max, _ = precision_recall_curve(labels, -losses.max(1)[0])
    precision_mean, recall_mean, _ = precision_recall_curve(labels, -losses.mean(1))
    metrics["AUPRC (avg)"] = auc(recall_mean, precision_mean)
    metrics["AUPRC (max)"] = auc(recall_max, precision_max)

    fpr_max, tpr_max, _ = roc_curve(labels, -losses.max(1)[0])
    fpr_mean, tpr_mean, _ = roc_curve(labels, -losses.mean(1))
    metrics["AUROC (avg)"] = auc(fpr_mean, tpr_mean)
    metrics["AUROC (max)"] = auc(fpr_max, tpr_max)
    return metrics


def setup(cfg):
    fabric = L.Fabric(**instantiate(cfg.trainer))
    fabric._loggers[0].log_hyperparams(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    L.seed_everything(cfg.seed, workers=True)
    fabric.launch()
    return fabric


def run_eval(cfg):
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fabric = setup(cfg)
    # technically it's possible to do multi-GPU eval, we just need to adjust
    # the shuffled_indices to be per-process
    # or account for mod % num_gpu from DistributedSampler's native sampling
    assert fabric.world_size == 1, "Only single GPU is supported"
    assert not (cfg.compute_metrics and "dense" in cfg.surprise)
    assert cfg.data.transform.crop_size == cfg.model.net.img_size
    log.info(f"{cfg.data.name}:{cfg.data.property}")

    # data
    dataset_kwargs, video_metadata = instantiate(cfg.data.data_fn)()
    transform = instantiate(cfg.data.transform)
    dataset = SlidingWindowVideoPredictionDataset(
        **dataset_kwargs,
        num_frames=cfg.model.net.num_frames,
        stride=cfg.stride,
        frame_step=cfg.data.frame_step,
        transform=transform,
        validate_end=True,
        # pred args
        img_size=cfg.data.transform.crop_size,
        patch_size=cfg.model.net.patch_size,
        tubelet_size=cfg.model.net.tubelet_size,
        context_length=cfg.context_length,
        mask_as_bool=cfg.mask_as_bool,
        first_sample_mode=cfg.data.first_sample_mode,
    )

    # validate data order, we need to ensure that data is match-ordered
    matches = torch.tensor([metadata["match"] for metadata in video_metadata])
    labels = torch.tensor([metadata["label"] for metadata in video_metadata])
    canon_order = (matches * 2 + labels).argsort()  # sort by (match, label)
    matches = matches[canon_order]
    labels = labels[canon_order]
    assert torch.all(matches[::2] == matches[1::2])
    assert torch.all(labels[::2] == 0) and torch.all(labels[1::2] == 1)

    # Initialize model
    with fabric.init_module(empty_init=True):
        net = instantiate(cfg.model.net)

    net = fabric.setup(net)

    if cfg.model.pretrained:
        net.load_ckpt(**cfg.model.ckpt_kwargs)
    net.freeze()

    log.info("Running eval")

    surprises = {
        "l1": l1_features,
        "l1_dense": l1_features_dense,
        "cross_entropy_sk": cross_entropy_sk,
        "cross_entropy_sk_dense": cross_entropy_sk_dense,
    }
    surprise = surprises[cfg.surprise]

    # little caveat: for mask_as_bool=False and start_frame_mode!='none',
    # we can't batch masks, so we'll have to do that separately.
    # the strategy is to first gather all "variable-length" masks (we can group by context_length)
    # create a loader for that
    # then for the remaining masks, we can batch them normally

    log.info("Dataset stats:")
    log.info(f"\tTotal videos: {len(dataset._num_samples)}")
    log.info(f"\tMedian #samples per video: {np.median(dataset._num_samples)}")
    groups = dataset.groupby_context_length()
    log.info("Group stats:")
    for ctx_len, indices in groups.items():
        log.info(f"\tContext length {ctx_len}: {len(indices)} samples")

    if dataset.mask_as_bool or cfg.data.first_sample_mode == "none":
        log.info("Disabling per-ctxsize batching. Batching all samples together.")
        groups = {dataset.context_length: list(range(len(dataset)))}

    all_losses, all_indices = [], []
    for gix, (_, indices) in enumerate(groups.items()):  # groups are already sorted by context length
        log.info(f"Processing group {gix} with {len(indices)} samples")
        # pre-shuffle indices for IID batching
        perm = torch.randperm(len(indices))
        shuffled_indices = [indices[i] for i in perm.tolist()]
        loader = DataLoader(
            Subset(dataset, shuffled_indices),
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
        loader = fabric.setup_dataloaders(loader)
        per_group_losses = compute_losses(net=net, loader=loader, surprise_fn=surprise)

        # unshuffle losses
        inv_indices = torch.argsort(perm)
        per_group_losses = per_group_losses[inv_indices]
        all_losses.append(per_group_losses)
        all_indices.append(torch.tensor(indices, dtype=torch.long))

    # concatenate and reorder all losses
    all_losses = torch.cat(all_losses, dim=0)
    all_indices = torch.cat(all_indices, dim=0)
    losses = torch.empty_like(all_losses)
    losses[all_indices] = all_losses
    # split into per-video losses
    losses = torch.split(losses, dataset.num_samples)
    # sort back into (match, label) order
    video_metadata = [video_metadata[i] for i in canon_order.tolist()]
    losses = [losses[i] for i in canon_order.tolist()]

    log.info("Computing metrics")

    torch.save(
        {
            "block": cfg.data.property,
            "frame_step": cfg.data.frame_step,
            "context_length": cfg.context_length,
            "losses": losses,
            "labels": labels,
            "matches": matches,
            "metadata": video_metadata,
        },
        output_dir / "losses.pth",
    )

    if cfg.compute_metrics:
        # padded losses do not affect metrics computation
        losses = pad_sequence(losses, batch_first=True, padding_value=0.0)
        assert losses.ndim == 2
        assert losses.shape[0] == labels.shape[0]
        # losses = [B, S]
        metrics = compute_metrics(losses, labels)
        fabric.log_dict(metrics)
        log.info(f"Metrics: {metrics}")


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    try:
        run_eval(cfg)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # this is important for hydra-submitit
        log.exception(e)
        raise e
    finally:
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()
