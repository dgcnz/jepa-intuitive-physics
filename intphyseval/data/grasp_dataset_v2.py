from pathlib import Path
import polars as pl

PROPERTIES = [
    "Collision",
    "Continuity",
    "Gravity",
    "GravityContinuity",
    "GravityInertia",
    "GravityInertia2",
    "GravitySupport",
    "Inertia",
    "Inertia2",
    "ObjectPermanence",
    "ObjectPermanence2",
    "ObjectPermanence3",
    "SolidityContinuity",
    "SolidityContinuity2",
    "Unchangeableness",
    "Unchangeableness2",
]


def get_videos_grasp(
    root: str,
    property: str,
    meta_path: str,
) -> tuple[dict, list[dict]]:
    """
    Get all videos from dataset.

    :param root: Path to dataset root
    :param property: Property to evaluate
    :param meta_path: Path to precomputed metadata CSV with scene pairings
    """
    root = Path(root)
    df = pl.read_csv(meta_path)
    df = df.filter(pl.col("property") == property)

    videos, metadata = [], []
    for row in df.iter_rows(named=True):
        videos.append(root / row["path"])
        metadata.append(
            {
                "label": row["label"],
                "match": row["match"],
            }
        )
    return {"videos": videos}, metadata
