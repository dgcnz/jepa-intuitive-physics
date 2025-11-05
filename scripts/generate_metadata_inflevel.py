import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd
from decord import VideoReader, cpu
from tqdm import tqdm


log = logging.getLogger("generate_metadata_inflevel")


def _scene_from_path(rel_path: str) -> str:
    # e.g., continuity/center__continuity__... -> scene 'center'
    stem = Path(rel_path).stem
    return stem.split("__")[0]


def main():
    parser = argparse.ArgumentParser(description="Generate InfLevel metadata (flatten pair CSVs)")
    parser.add_argument("--root", type=str, required=True, help="Path to InfLevel dataset root")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("intphyseval/data/metadata/inflevel/meta.csv")),
        help="Canonical output CSV path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    root = Path(args.root)
    out_path = Path(args.output)

    # Where the original pair CSVs live
    meta_dir = Path("intphyseval/data/metadata/inflevel")
    assert meta_dir.exists(), f"Missing metadata directory: {meta_dir}"

    csv_files = sorted([p for p in meta_dir.iterdir() if p.suffix == ".csv" and p.name not in {"meta.csv"}])
    assert csv_files, f"No property CSVs found under {meta_dir}"

    rows: List[dict] = []
    for csv_path in tqdm(csv_files, desc="Processing properties"):
        prop = csv_path.stem
        df = pd.read_csv(csv_path)

        # preserve original row order; match = row index
        for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {prop}", leave=False):
            # vid1 as video=0
            v1_rel = row["vid1_path"]
            v1_abs = root / v1_rel
            assert v1_abs.exists(), f"Missing video: {v1_abs}"
            v1_frames = len(VideoReader(str(v1_abs), num_threads=-1, ctx=cpu(0)))
            rows.append(
                {
                    "property": prop,
                    "scene": _scene_from_path(v1_rel),
                    "video": 0,
                    "match": row_idx,
                    "label": 1 if row["vid1_label"] == "possible" else 0,
                    "path": str(v1_rel),
                    "frames": v1_frames,
                    "format": "mp4",
                    "start_priming_1": int(row["vid1_start_priming_1"]),
                    "end_priming_1": int(row["vid1_end_priming_1"]),
                    "start_priming_2": int(row["vid1_start_priming_2"]),
                    "end_priming_2": int(row["vid1_end_priming_2"]),
                }
            )

            # vid2 as video=1
            v2_rel = row["vid2_path"]
            v2_abs = root / v2_rel
            assert v2_abs.exists(), f"Missing video: {v2_abs}"
            v2_frames = len(VideoReader(str(v2_abs), num_threads=-1, ctx=cpu(0)))
            rows.append(
                {
                    "property": prop,
                    "scene": _scene_from_path(v2_rel),
                    "video": 1,
                    "match": row_idx,
                    "label": 1 if row["vid2_label"] == "possible" else 0,
                    "path": str(v2_rel),
                    "frames": v2_frames,
                    "format": "mp4",
                    "start_priming_1": int(row["vid2_start_priming_1"]),
                    "end_priming_1": int(row["vid2_end_priming_1"]),
                    "start_priming_2": int(row["vid2_start_priming_2"]),
                    "end_priming_2": int(row["vid2_end_priming_2"]),
                }
            )

    # Ordering: by property ASC; within property preserve the CSV row order; within each pair emit video 0 then 1
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["property"]).reset_index(drop=True)

    # Validations
    assert (df["frames"] > 0).all(), "All InfLevel videos must have >0 frames"
    assert set(df["format"]) == {"mp4"}, "InfLevel format must be mp4"
    assert df[["property", "match", "video"]].drop_duplicates().shape[0] == df.shape[0], "(property, match, video) must be unique"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()

