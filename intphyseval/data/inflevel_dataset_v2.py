from pathlib import Path
import polars as pl

PROPERTIES = ["continuity", "gravity", "solidity"]


def get_videos_inflevel(
    root: str,
    property: str,
    meta_path: str,
) -> tuple[dict, list[dict]]:
    root = Path(root)
    df = pl.read_csv(meta_path)
    df = df.filter(pl.col("property") == property)

    videos, metadata = [], []
    start, end = [], []
    for _, group in df.group_by("match", maintain_order=True):
        assert len(group) == 2, "Expected exactly two videos per match"
        # we need to sample start frames so that both clips have the same length
        end_0 = group.row(0)["frames"]
        end_1 = group.row(1)["frames"]
        start_0 = group.row(0)["end_priming_2"] + 1
        start_1 = group.row(1)["end_priming_2"] + 1
        common_len = min(end_0 - start_0, end_1 - start_1)
        start_0 = end_0 - common_len
        start_1 = end_1 - common_len

        videos.append(root / group.row(0)["path"])
        start.append(start_0)
        end.append(end_0)
        metadata.append(
            {
                "label": group.row(0)["label"],
                "match": group.row(0)["match"],
            }
        )

        videos.append(root / group.row(1)["path"])
        start.append(start_1)
        end.append(end_1)
        metadata.append(
            {
                "label": group.row(1)["label"],
                "match": group.row(1)["match"],
            }
        )
    return {"videos": videos, "start": start, "end": end}, metadata
