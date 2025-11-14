import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence, Callable

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from aquarel import Theme
from hydra.utils import instantiate
from jaxtyping import Float, Int
from omegaconf import OmegaConf
from scipy.special import softmax
from skimage.transform import resize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from tqdm import tqdm

from intphyseval.data.sliding_window_dataset import SlidingWindowVideoDataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MEAN_255 = np.asarray(IMAGENET_DEFAULT_MEAN, dtype=np.float32).reshape(3, 1, 1) * 255.0
STD_255 = np.asarray(IMAGENET_DEFAULT_STD, dtype=np.float32).reshape(3, 1, 1) * 255.0

COLORS = {
    "red": "#FF3B30",
    "blue": "#007AFF",
    "green": "#34C759",
    "purple": "#AF52DE",
    "black": "#1D1D1F",
}

MPL_THEME = (
    Theme(name="custom", description="")
    .set_font(family="Source Sans Pro")
    .set_grid(draw=True, width=0.4)
    .set_color(
        palette=[
            COLORS["blue"],
            COLORS["red"],
            COLORS["purple"],
            COLORS["green"],
        ],
        grid_color="#E5E5EA",
    )
)


def denormalize_frames(frames: np.ndarray) -> np.ndarray:
    return np.clip((frames * STD_255 + MEAN_255) / 255.0, 0.0, 1.0)


def render_heatmaps(
    heatmaps: np.ndarray,
    *,
    shape: tuple[int, int],
    yrange: tuple[float, float],
    cmap: str,
) -> np.ndarray:
    heatmaps = heatmaps.astype(np.float32)
    H, W = shape
    vmin, vmax = yrange
    heat_upsampled = resize(
        heatmaps,
        (heatmaps.shape[0], H, W),
        order=1,
        mode="reflect",
        anti_aliasing=False,
        preserve_range=True,
    )
    heat_norm = np.clip((heat_upsampled - vmin) / (vmax - vmin), 0.0, 1.0)
    heat_rgb = mpl.colormaps[cmap](heat_norm)[..., :3].astype(np.float32)
    return np.transpose(heat_rgb, (0, 3, 1, 2))


def overlay_heatmaps(
    frames_chw: np.ndarray,
    heatmaps_hw: np.ndarray,
    *,
    shape: tuple[int, int],
    yrange: tuple[float, float],
    alpha: float = 0.5,
    cmap: str = "magma",
) -> np.ndarray:
    frames_denorm = denormalize_frames(frames_chw)
    heat_rgb = render_heatmaps(heatmaps_hw, shape=shape, yrange=yrange, cmap=cmap)
    overlays = np.clip((1 - alpha) * frames_denorm + alpha * heat_rgb, 0.0, 1.0)
    return overlays


def losses_to_grid(
    pair_losses: Float[Tensor, "2 S N"],
    img_size: int,
    patch_size: int,
    num_frames: int,
    tubelet_size: int,
    context_len: int,
) -> Float[Tensor, "2 S P' H' W'"]:
    H_grid = W_grid = img_size // patch_size
    # T_prime, C_prime, = [f // tubelet_size for f in [num_frames, ctx_len]]
    P_prime = (num_frames - context_len) // tubelet_size
    N_expected = P_prime * H_grid * W_grid
    assert pair_losses.shape[2] == N_expected
    return pair_losses.view(2, -1, P_prime, H_grid, W_grid)


def aggregate_losses_to_frames_v3(
    loss_grid: Float[Tensor, "2 S P' H' W'"],
    *,
    stride: int,
    context_length: int,
    tubelet_size: int,
    num_frames: int,
) -> Float[Tensor, "2 Tp H' W'"]:
    """
    Aggregate tubelet-level losses to per-frame heatmaps using torch.fold.

    Optimized version for when stride is divisible by tubelet_size.
    Works at tubelet resolution, then expands to frame level.
    """
    assert all(
        [
            stride % tubelet_size == 0,
            num_frames % tubelet_size == 0,
            context_length % tubelet_size == 0,
        ]
    ), f"ts={tubelet_size} must divide s={stride}, nf={num_frames}, cl={context_length}"

    _, S, P_prime, H_grid, W_grid = loss_grid.shape
    losses_for_fold = loss_grid.permute(0, 3, 4, 2, 1).flatten(0, 2)
    stride_tubelet = stride // tubelet_size
    output_size = (S - 1) * stride_tubelet + P_prime

    heatmaps_tub = torch.nn.functional.fold(
        losses_for_fold,
        output_size=(output_size, 1),
        kernel_size=(P_prime, 1),
        stride=(stride_tubelet, 1),
    )
    counts_tub = torch.nn.functional.fold(
        torch.ones_like(losses_for_fold),
        output_size=(output_size, 1),
        kernel_size=(P_prime, 1),
        stride=(stride_tubelet, 1),
    )

    heatmaps_tub = heatmaps_tub[:, 0, :, 0]
    counts_tub = counts_tub[:, 0, :, 0]

    heatmaps_tub = heatmaps_tub.unflatten(0, (2, H_grid, W_grid)).permute(0, 3, 1, 2)
    counts_tub = counts_tub.unflatten(0, (2, H_grid, W_grid)).permute(0, 3, 1, 2)
    assert (counts_tub > 0).all(), "counts must be nonzero"

    heatmaps_tub = heatmaps_tub / counts_tub
    return heatmaps_tub.repeat_interleave(tubelet_size, dim=1)


class AnimatedPlot(object):
    def __init__(self, ax):
        self.ax = ax

    def update(self, frame_idx: int):
        raise NotImplementedError


class VideoPlot(AnimatedPlot):
    def __init__(self, ax, title: str, frames: Sequence[np.ndarray]):
        super().__init__(ax)
        self.frames = list(frames)
        self.image_artist = ax.imshow(self.frames[0])
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    def update(self, frame_idx: int):
        self.image_artist.set_data(self.frames[frame_idx])


class LinePlot(AnimatedPlot):
    def __init__(
        self,
        ax,
        *,
        title: str,
        xs: Int[np.ndarray, "N"],  # noqa
        ys: Float[np.ndarray, "N"],  # noqa
        color: str,
        ylabel: str,
        ylim: tuple[float, float],
    ):
        super().__init__(ax)
        self.xs, self.ys = xs, ys
        (self.line,) = ax.plot([], [], color=color, linewidth=2)

        ax.set(xlim=(self.xs[0], self.xs[-1]), ylim=ylim)
        ax.set_xlabel("Frame")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=11)

    def update(self, pred_idx: int):
        self.line.set_data(self.xs[: pred_idx + 1], self.ys[: pred_idx + 1])


class ShadedLinePlot(LinePlot):
    """Line plot that fills areas above/below y0 with configurable colors."""

    def __init__(
        self,
        ax,
        *,
        y0: float = 0.0,
        colors: tuple[str, str] = (COLORS["red"], COLORS["blue"]),
        **kwargs,
    ):
        super().__init__(ax, **kwargs)
        self.y0 = y0
        self.fill_positive_color, self.fill_negative_color = colors
        self.fill_collections = []
        ax.axhline(y0, color="gray", linestyle="-", linewidth=0.5)

    def update(self, frame_idx: int):
        super().update(frame_idx)
        for coll in self.fill_collections:
            coll.remove()
        self.fill_collections.clear()
        if frame_idx == 0:
            return

        ys = self.ys[: frame_idx + 1]
        self.fill_collections.extend(
            self.ax.fill_between(
                self.xs[: frame_idx + 1],
                self.y0,
                ys,
                where=where,
                color=color,
                alpha=0.3,
                interpolate=True,
            )
            for where, color in [
                (ys >= self.y0, self.fill_positive_color),
                (ys < self.y0, self.fill_negative_color),
            ]
        )


class PairAnimationFigure:
    def __init__(
        self,
        match_id: int,
        clips: Float[np.ndarray, "2 3 T H W"],  # T = total clip length
        heatmaps: Float[np.ndarray, "2 Tp h w"],  # Tp = pred frames len, h = H / P, ...
        avg_losses: Float[np.ndarray, "2"],
        frame_indices: Int[np.ndarray, "Tp"],  # noqa
        tags: list[str],
        surprise_name: str,
        tau: Optional[float],
    ):
        self.match_id = match_id
        self.frame_indices = frame_indices
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.lines = self._lineplots(heatmaps, avg_losses)
        self.images = self._heatmaps(heatmaps, clips, tau)
        self.framecnt = self._titles(avg_losses, tags, surprise_name, tau)[-1]

    def _lineplots(
        self,
        heatmaps: Float[np.ndarray, "2 Tp h w"],
        avg_losses: Float[np.ndarray, "2"],
    ):
        # compute temporal surprise with joint y-limits
        frame_losses = heatmaps.mean(axis=(-2, -1))
        ylo, yhi = frame_losses.min(), frame_losses.max()
        pad = (yhi - ylo) * 0.1
        ylo, yhi = ylo - pad, yhi + pad

        # compute difference temporal surprise with symmetric y-limits
        frame_loss_diff = frame_losses[0] - frame_losses[1]
        diff_abs = 1.1 * float(np.max(np.abs(frame_loss_diff)))

        labels, colors = ["Imp", "Pos"], [COLORS["red"], COLORS["blue"]]
        return [
            LinePlot(
                self.axes[1, i],
                title=f"Surprise over time ({labels[i]}, Avg: {avg_losses[i]:.3g})",
                xs=self.frame_indices,
                ys=frame_losses[i],
                color=colors[i],
                ylabel="Surprise",
                ylim=(ylo, yhi),
            )
            for i in range(2)
        ] + [
            ShadedLinePlot(
                self.axes[1, 2],
                title="Relative Surprise over time (Imp - Pos)",
                xs=self.frame_indices,
                ys=frame_loss_diff,
                color=COLORS["purple"],
                ylabel="Relative Surprise",
                ylim=(-diff_abs, diff_abs),
                y0=0.0,
                colors=colors,
            )
        ]

    def _heatmaps(
        self,
        heatmaps: Float[np.ndarray, "2 Tp h w"],
        clips: Float[np.ndarray, "2 3 T H W"],
        tau: Optional[float],
    ):
        # normalize heatmaps if needed + compute relative heatmap
        if tau is not None:
            heatmaps = softmax(heatmaps.reshape(-1) / tau, axis=0).reshape(heatmaps.shape)  # fmt: off
        diff_heatmaps = heatmaps[0] - heatmaps[1]

        # joint vmin/vmax for overlays, symmetric for relative
        vmax_diff = float(np.max(np.abs(diff_heatmaps)))
        vmin, vmax = float(heatmaps.min()), float(heatmaps.max())
        assert vmin < vmax, "Invalid heatmap range"

        # the first `context_length`` frames don't have predictions
        valid_clip = clips[:, :, self.frame_indices, :, :].transpose(0, 2, 1, 3, 4)

        H, W = clips.shape[-2:]
        overlays = overlay_heatmaps(
            valid_clip.reshape(-1, *valid_clip.shape[2:]),
            heatmaps.reshape(-1, *heatmaps.shape[2:]),
            shape=(H, W),
            yrange=(vmin, vmax),
        ).reshape(2, -1, 3, H, W)
        rel_frames = render_heatmaps(
            diff_heatmaps,
            shape=(H, W),
            yrange=(-vmax_diff, vmax_diff),
            cmap="RdBu_r",
        )
        overlays = np.moveaxis(overlays, 2, -1)  # to NHWC for display
        rel_frames = np.moveaxis(rel_frames, 1, -1)  # to NHWC for display
        return [
            VideoPlot(self.axes[0, 0], "Surprise Heatmap (Imp)", overlays[0]),
            VideoPlot(self.axes[0, 1], "Surprise Heatmap (Pos)", overlays[1]),
            VideoPlot(
                self.axes[0, 2], "Relative Surprise Heatmap (Imp - Pos)", rel_frames
            ),
        ]

    def _titles(
        self,
        avg_losses: Float[np.ndarray, "2"],
        tags: list[str],
        surprise_name: str,
        tau: Optional[float],
    ):
        correct = avg_losses[0] > avg_losses[1]
        title = [f"match {self.match_id}"] + tags + [surprise_name]
        if tau is not None:
            title.append(f"τ: {tau}")
        title = "  |  ".join(title)

        black, green, red = COLORS["black"], COLORS["green"], COLORS["red"]
        symbol, color = ("✓", green) if correct else ("x", red)
        self.fig.suptitle(title, fontsize=20, color=black, x=0.5, y=0.98)
        kwargs = dict(y=0.98, fontsize=18, va="top", transform=self.fig.transFigure)
        t, n = self.frame_indices[0], self.frame_indices[-1]
        return [
            self.fig.text(x=x, s=s, color=c, ha=ha, **kwargs)
            for x, s, c, ha in [
                (0.01, f"[{symbol}]", color, "left"),
                (0.98, f"Frame: {t}/{n}", black, "right"),
            ]
        ]

    def update(self, idx: int):
        t = self.frame_indices[idx]

        self.framecnt.set_text(f"Frame: {t}/{self.frame_indices[-1]}")
        for plot in self.images + self.lines:
            plot.update(idx)

    def render(self, *, out_path: Path, fps: int):
        self.fig.tight_layout()
        with MPL_THEME:
            ani = animation.FuncAnimation(
                self.fig,
                self.update,
                frames=len(self.frame_indices),
                interval=100,
                blit=False,
            )
        writer = animation.PillowWriter(fps=fps)
        ani.save(out_path, writer=writer, dpi=300)
        plt.close(self.fig)
        log.info(f"Saved {out_path}")


def run(
    load_clip: Callable,
    losses: list[Tensor],
    matches: Tensor,
    labels: Tensor,
    surprise_name: str,
    out_dir: Path,
    fps: int,
    tau: Optional[float],
    max_pairs: Optional[int],
    tags: list[str],
    stride: int,
    img_size: int,
    patch_size: int,
    tubelet_size: int,
    num_frames: int,
    context_len: int,
):
    total_pairs = len(losses) // 2
    max_pairs = min(max_pairs or total_pairs, total_pairs)
    assert total_pairs * 2 == len(losses)
    assert len(losses) == len(matches) == len(labels)

    for pix in tqdm(range(max_pairs), total=max_pairs):
        i0, i1 = pix * 2, pix * 2 + 1
        match_id = matches[i0].item()
        raw_losses = torch.stack([losses[i0], losses[i1]], dim=0)

        loss_grid = losses_to_grid(
            raw_losses,
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            context_len=context_len,
        )
        heatmaps = aggregate_losses_to_frames_v3(
            loss_grid,
            stride=stride,
            context_length=context_len,
            tubelet_size=tubelet_size,
            num_frames=num_frames,
        )
        clips = torch.stack([load_clip(i0), load_clip(i1)], axis=0)
        frame_indices = np.arange(context_len, clips.shape[2])

        animator = PairAnimationFigure(
            match_id=match_id,
            clips=clips.numpy(),
            heatmaps=heatmaps.numpy(),
            avg_losses=raw_losses.numpy().mean((1, 2)),
            frame_indices=frame_indices,
            tags=tags,
            surprise_name=surprise_name,
            tau=tau,
        )
        animator.render(
            out_path=out_dir / f"match_{match_id:04d}.webp",
            fps=fps,
        )


def load_dataset(cfg):
    dataset_kwargs, video_metadata = instantiate(cfg.data.data_fn)()
    matches = torch.tensor([metadata["match"] for metadata in video_metadata])
    labels = torch.tensor([metadata["label"] for metadata in video_metadata])
    canon_order = (matches * 2 + labels).argsort()
    matches = matches[canon_order]
    labels = labels[canon_order]

    transform = instantiate(cfg.data.dataloader.dataset.transform)
    videos = dataset_kwargs["videos"]
    assert "start_frames" not in dataset_kwargs, "not supported"
    assert "end_frames" not in dataset_kwargs, "not supported"
    # assert that match_idx[i*2] and match_idx[i*2+1] refer to the same video vectorized
    assert torch.equal(matches[::2], matches[1::2])

    videos = [videos[i] for i in canon_order]

    dataset = SlidingWindowVideoDataset(
        videos=videos,
        num_frames=cfg.model.net.num_frames,
        stride=cfg.stride,
        frame_step=cfg.data.frame_step,
        transform=transform,
        validate_end=False,
    )
    return dataset, matches, labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render heatmap overlays from token-level JEPA losses (OOP v4)"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory with losses.pth and .hydra/config.yaml",
    )
    parser.add_argument("--fps", type=int, default=5, help="Animation FPS")
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Temperature for softmax normalization of heatmaps (optional). Lower = more peaked, higher = more uniform.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Maximum number of pairs to process (default: process all)",
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = (
        Path(__file__).parent / "outputs" / run_dir.name / f"tau_{args.tau or 'none'}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(run_dir / "losses.pth", map_location="cpu")
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    dataset, matches, labels = load_dataset(cfg)

    assert cfg.tags is not None and len(cfg.tags) > 0, (
        "cfg.tags must be set for display purposes"
    )
    tags = list(cfg.tags)
    losses = [torch.as_tensor(loss) for loss in payload["losses"]]
    payload_labels = torch.as_tensor(payload["labels"])

    assert len(losses) == len(matches) == len(labels) == len(payload_labels)
    assert torch.equal(labels.cpu(), payload_labels.cpu()), "Label ordering mismatch"
    assert cfg.surprise

    load_clip = lambda idx: dataset._load_clip(idx, with_tx=True)  # noqa
    run(
        load_clip=load_clip,
        losses=losses,
        matches=matches,
        labels=labels,
        surprise_name=cfg.surprise,
        out_dir=out_dir,
        fps=args.fps,
        tau=args.tau,
        max_pairs=args.max_pairs,
        tags=tags,
        stride=cfg.stride,
        img_size=cfg.model.net.img_size,
        patch_size=cfg.model.net.patch_size,
        tubelet_size=cfg.model.net.encoder.tubelet_size,
        num_frames=cfg.model.net.num_frames,
        context_len=cfg.context_length,
    )


if __name__ == "__main__":
    main()
