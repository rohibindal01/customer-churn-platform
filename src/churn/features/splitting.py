"""Stratified train / validation / test split and its saved copies."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config.settings import Settings, get_settings
from churn.utils.logger import get_logger

logger = get_logger(__name__)

SPLIT_NAMES = ("train", "validation", "test")


def make_target(df: pd.DataFrame, settings: Settings | None = None) -> pd.Series:
    """Churn as 1 (positive label) / 0."""
    settings = settings or get_settings()
    labels = df[settings.data.target_column].astype("string").str.strip()
    flag = (labels == settings.data.positive_label).fillna(False)
    return flag.astype(int).rename("churn")


def split_xy(
    df: pd.DataFrame, settings: Settings | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate raw features (no ID, no target) from the 0/1 target."""
    settings = settings or get_settings()
    drop = [
        c for c in (settings.data.id_column, settings.data.target_column)
        if c in df.columns
    ]
    return df.drop(columns=drop), make_target(df, settings)


def stratified_split(
    df: pd.DataFrame, settings: Settings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split raw rows into train / validation / test with equal churn rates.

    Step 1 takes the test share off. Step 2 splits the rest into train and
    validation, using the share that makes the final sizes match config.yaml.
    """
    settings = settings or get_settings()
    cfg = settings.split
    df = df.reset_index(drop=True)
    y = make_target(df, settings)

    train_val, test = train_test_split(
        df, test_size=cfg.test_size, stratify=y, random_state=settings.random_seed
    )
    val_fraction = cfg.validation_size / (cfg.train_size + cfg.validation_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=y.loc[train_val.index],
        random_state=settings.random_seed,
    )
    train, val, test = (f.reset_index(drop=True) for f in (train, val, test))
    logger.info("Split sizes: train=%d, validation=%d, test=%d",
                len(train), len(val), len(test))
    return train, val, test


def save_splits(
    splits: dict[str, pd.DataFrame],
    out_dir: Path | str | None = None,
    extra: dict | None = None,
    settings: Settings | None = None,
) -> dict:
    """Write each split to CSV plus split_manifest.json; return the manifest."""
    settings = settings or get_settings()
    directory = Path(out_dir) if out_dir else settings.processed_dir
    directory.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "random_seed": settings.random_seed,
        "configured_sizes": {
            "train": settings.split.train_size,
            "validation": settings.split.validation_size,
            "test": settings.split.test_size,
        },
        "splits": {},
    }
    for name, frame in splits.items():
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False)
        y = make_target(frame, settings)
        manifest["splits"][name] = {
            "file": path.name,
            "rows": int(len(frame)),
            "churned": int(y.sum()),
            "churn_rate_pct": round(float(y.mean() * 100), 2),
        }
    if extra:
        manifest.update(extra)

    (directory / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def load_split(
    name: str,
    directory: Path | str | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Read one saved split ('train', 'validation' or 'test')."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split '{name}'. Choose from {SPLIT_NAMES}.")
    settings = settings or get_settings()
    path = (Path(directory) if directory else settings.processed_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run: python scripts\\03_prepare_data.py"
        )
    return pd.read_csv(path)