import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd
from decord import VideoReader, cpu


log = logging.getLogger("generate_metadata_grasp")


def main():
    parser = argparse.ArgumentParser(description="Generate GRASP metadata (video-per-row)")
    parser.add_argument("--root", type=str, required=True, help="Path to GRASP root with P_*/IP_* subfolders")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("intphyseval/data/metadata/grasp/meta.csv")),
        help="Canonical output CSV path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    root = Path(args.root)
    out_path = Path(args.output)

    # Discover properties
    props = sorted([d.name[2:] for d in root.iterdir() if d.is_dir() and d.name.startswith("P_")])
    assert props, f"No P_* properties found under {root}"

    rows: List[dict] = []
    for prop in props:
        p_dir = root / f"P_{prop}"
        ip_dir = root / f"IP_{prop}"
        assert ip_dir.exists(), f"Missing IP_{prop} for property {prop}"

        p_files = sorted([f.name for f in p_dir.iterdir() if f.is_file() and f.suffix.lower() == ".mp4"])
        ip_files = sorted([f.name for f in ip_dir.iterdir() if f.is_file() and f.suffix.lower() == ".mp4"])
        assert p_files == ip_files, f"Basename mismatch between P_{prop} and IP_{prop}"

        for scene_idx, fname in enumerate(p_files):
            scene_stem = Path(fname).stem

            # Possible video (video=0)
            p_rel = Path(f"P_{prop}") / fname
            p_path = root / p_rel
            p_frames = len(VideoReader(str(p_path), num_threads=-1, ctx=cpu(0)))
            rows.append(
                {
                    "property": prop,
                    "scene": scene_stem,
                    "video": 0,
                    "match": scene_idx,
                    "label": 1,
                    "path": str(p_rel),
                    "frames": p_frames,
                    "format": "mp4",
                }
            )

            # Impossible video (video=1)
            ip_rel = Path(f"IP_{prop}") / fname
            ip_path = root / ip_rel
            ip_frames = len(VideoReader(str(ip_path), num_threads=-1, ctx=cpu(0)))
            rows.append(
                {
                    "property": prop,
                    "scene": scene_stem,
                    "video": 1,
                    "match": scene_idx,
                    "label": 0,
                    "path": str(ip_rel),
                    "frames": ip_frames,
                    "format": "mp4",
                }
            )

    # Sort: property ASC, then scene (lexicographic), then video ASC
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["property", "scene", "video"]).reset_index(drop=True)

    # Validations
    assert (df["frames"] > 0).all(), "All GRASP videos must have >0 frames"
    assert set(df["format"]) == {"mp4"}, "GRASP format must be mp4"
    assert df[["property", "match", "video"]].drop_duplicates().shape[0] == df.shape[0], "(property, match, video) must be unique"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()

