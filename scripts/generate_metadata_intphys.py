import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from intphyseval.utils import get_breaking_points, get_matches


log = logging.getLogger("generate_metadata_intphys")


def _infer_pairs_for_scene(root: Path, prop: str, scene: str) -> Tuple[List[List[int]], List[int]]:
    """Load the 4 videos for a scene and infer the two matching pairs.

    Returns (matches, labels), where:
      - matches: list of two pairs of 0-based indices, e.g., [[0,1],[2,3]]
      - labels: list of length 4 with 1 for possible, 0 for impossible
    """
    vids = []  # list of [C, T, H, W]
    labels: List[int] = []
    for vid_id in (1, 2, 3, 4):
        base = root / prop / scene / str(vid_id)
        frame_dir = base / "scene"
        status_path = base / "status.json"

        if not frame_dir.exists() or not status_path.exists():
            raise FileNotFoundError(f"Missing required files under {base}")

        frame_files = sorted([p for p in frame_dir.iterdir() if p.is_file()])
        assert frame_files, f"No frames in {frame_dir}"

        sampled = frame_files[0::2]
        assert sampled, f"No frames at step=2 in {frame_dir}"
        thwc = torch.stack([torch.Tensor(np.array(Image.open(p))) for p in sampled], dim=0)
        vid = thwc.permute(3, 0, 1, 2).contiguous()  # C T H W
        vids.append(vid)

        with open(status_path, "r") as f:
            status = json.load(f)
        labels.append(1 if status["header"]["is_possible"] else 0)

    min_T = min(v.shape[1] for v in vids)
    vids = [v[:, :min_T] for v in vids]
    clip = torch.stack(vids, dim=0)  # [4, C, T, H, W]

    bps = get_breaking_points(clip)
    pairs = get_matches(bps)
    return pairs, labels


def _detect_format_and_count(frame_dir: Path) -> Tuple[str, int]:
    files = sorted([p for p in frame_dir.iterdir() if p.is_file()])
    assert files, f"No files found in {frame_dir}"

    exts = {f.suffix.lower() for f in files}
    assert len(exts) == 1, f"Mixed file extensions in {frame_dir}: {exts}"

    fmt = exts.pop().lstrip(".")
    return fmt, len(files)


def main():
    parser = argparse.ArgumentParser(description="Generate IntPhys metadata (video-per-row)")
    parser.add_argument("--root", type=str, required=True, help="Path to IntPhys dev root containing O1/O2/O3")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("intphyseval/data/metadata/intphys/dev/meta.csv")),
        help="Canonical output CSV path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    root = Path(args.root)
    out_path = Path(args.output)

    rows: List[dict] = []
    for prop in sorted(["O1", "O2", "O3"]):
        prop_dir = root / prop
        scenes = sorted([d.name for d in prop_dir.iterdir() if d.is_dir()])
        for scene in tqdm(scenes, desc=f"Processing {prop}"):
            pairs, labels = _infer_pairs_for_scene(root, prop, scene)

            # For match uniqueness: (int(scene) - 1) * 2 + pair
            for pair_id, pair in enumerate(pairs):
                match_id = (int(scene) - 1) * 2 + pair_id
                for vid_idx in sorted(pair):
                    base = Path(prop) / scene / str(vid_idx + 1) / "scene"
                    frame_dir = root / base
                    fmt, n_frames = _detect_format_and_count(frame_dir)
                    rows.append(
                        {
                            "property": prop,
                            "scene": scene,
                            "video": vid_idx,
                            "match": match_id,
                            "pair": pair_id,
                            "label": labels[vid_idx],
                            "path": str(base),
                            "frames": n_frames,
                            "format": fmt,
                        }
                    )

    # Sort rows: property ASC, scene lexicographic ASC, video ASC
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["property", "scene", "video"]).reset_index(drop=True)

    # Final validations
    assert (df["frames"] > 0).all(), "All videos must have >0 frames"
    assert set(df["format"]) <= {"png", "jpg"}, "Unexpected formats in IntPhys"
    assert df[["property", "match", "video"]].drop_duplicates().shape[0] == df.shape[0], "(property, match, video) must be unique"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()

