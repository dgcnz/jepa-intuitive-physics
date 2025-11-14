from pathlib import Path
import polars as pl


SPLITS = ("Debug", "Main", "HeldOut")
CONDITIONS = ("continuity", "immutability", "permanence", "solidity")


def get_videos_intphys2(
    root: str,
    condition: str,
    split: str = "Main",
) -> tuple[dict, list[dict]]:
    """
    Prepare IntPhys2 videos and metadata for SlidingWindowVideoDataset.

    :param root: Dataset root directory containing Debug/Main/HeldOut folders.
    :param condition: Physics condition filter(s).
    :param split: Which split to load ("Debug", "Main", "HeldOut").
    :return: (dataset_kwargs, metadata_list)
    """
    split = split.capitalize()
    assert split in SPLITS, f"Unknown split '{split}'. Expected {SPLITS}."
    assert condition in CONDITIONS, f"Unknown '{condition}'. Exp. {CONDITIONS}."

    root_path = Path(root)
    meta_path = root_path / split / "metadata.csv"
    assert meta_path.exists(), f"metadata not found at {meta_path}"

    df = pl.read_csv(meta_path, schema_overrides={"SceneIndex": pl.String}) 
    # SceneIndex is ordered.
    # is almost an int: 0, 1, ..., 123, 124, 124_0, 124_1 , ...
    # since it's already sorted, and there are 4 videos per scene
    # the "normalized" scene idx would just be the video row index // 4
    df = df.filter(pl.col("condition") == condition)

    videos: list[str] = []
    metadata: list[dict] = []

    for row_ix, row in enumerate(df.iter_rows(named=True)):
        video_path = root_path / split / row["file_name"]
        videos.append(str(video_path))

        # type: 1_Possible, 1_Impossible, 2_Possible, 2_Impossible
        match_id = (row_ix // 4) * 2 + (int(row["type"][0]) - 1)
        metadata.append(
            {
                "label": int(row["type"].endswith("Possible")),
                "match": match_id,
                "scene_index": row["SceneIndex"],
                "type": row["type"],
                "difficulty": row["Difficulty"],
                "condition": row["condition"],
                "camera": row["Camera"],
                "environment": row["env"],
                "video_name": row["name"],
            }
        )

    return {"videos": videos}, metadata
