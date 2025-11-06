import polars as pl
from pathlib import Path

PROPERTIES = ["O1", "O2", "O3"]

def get_videos_intphys(
    root: str,
    property: str,  # O1
    meta_path: str,  # intphyseval/data/metadata/intphys/dev/meta.csv
) -> tuple[dict, list[dict]]:
    """
    Get all videos from dataset.

    :param root: Path to dataset root
    :param property: Property to evaluate (e.g., "O1", "O2", "O3")
    :param meta_path: Path to precomputed metadata CSV with scene pairings
    """
    root = Path(root)
    df = pl.read_csv(meta_path)
    df = df.filter(pl.col("property") == property)

    videos, metadata = [], []
    for row in df.iter_rows(named=True):
        frame_paths = list(sorted((root / row["path"]).iterdir()))
        videos.append(frame_paths)
        metadata.append(
            {
                "label": row["label"],
                "match": row["match"],
            }
        )
    return {"videos": videos}, metadata
