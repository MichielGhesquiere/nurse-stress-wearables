from pathlib import Path
from typing import List
import pandas as pd


def feature_path(root: str, feature_set: str, subject_id: str, date_str: str) -> Path:
    return Path(root) / f"feature_set={feature_set}" / f"id={subject_id}" / f"d={date_str}" / "part.parquet"


def write_features_per_day(features: pd.DataFrame, root: str, feature_set: str = "v1", force: bool = False) -> List[str]:
    """Write features partitioned by id and day; returns list of written paths."""
    if 'datetime' not in features.columns:
        raise ValueError("features requires 'datetime' column")
    feats = features.copy()
    # Ensure id is string to avoid parquet dtype conversion issues
    if 'id' in feats.columns:
        feats['id'] = feats['id'].astype(str)
    feats['d'] = pd.to_datetime(feats['datetime']).dt.date.astype(str)
    subject = str(feats['id'].iloc[0])
    written = []
    for d, part in feats.groupby('d', sort=True):
        out_path = feature_path(root, feature_set, subject, d)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not force:
            continue
        part.drop(columns=['d'], errors='ignore').to_parquet(out_path, index=False)
        written.append(str(out_path))
    return written
