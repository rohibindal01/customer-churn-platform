"""Tests for the stratified split and the saved split files."""
import json

import pytest

from churn.features.splitting import (
    load_split,
    make_target,
    save_splits,
    split_xy,
    stratified_split,
)


def test_sizes_follow_config(synthetic_df):
    train, val, test = stratified_split(synthetic_df)
    assert abs(len(train) - 240) <= 1
    assert abs(len(val) - 80) <= 1
    assert abs(len(test) - 80) <= 1


def test_splits_are_disjoint_and_complete(synthetic_df):
    train, val, test = stratified_split(synthetic_df)
    ids = [set(f["customerID"]) for f in (train, val, test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert sum(len(i) for i in ids) == len(synthetic_df)


def test_churn_rate_is_preserved_in_every_split(synthetic_df):
    overall = make_target(synthetic_df).mean()
    for frame in stratified_split(synthetic_df):
        assert make_target(frame).mean() == pytest.approx(overall, abs=0.02)


def test_split_is_reproducible(synthetic_df):
    first = stratified_split(synthetic_df)
    second = stratified_split(synthetic_df)
    for a, b in zip(first, second):
        assert list(a["customerID"]) == list(b["customerID"])


def test_split_xy_removes_id_and_target(synthetic_df):
    X, y = split_xy(synthetic_df)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert set(y.unique()) == {0, 1}
    assert len(X) == len(y) == len(synthetic_df)


def test_save_and_load_round_trip(synthetic_df, tmp_path):
    train, val, test = stratified_split(synthetic_df)
    manifest = save_splits(
        {"train": train, "validation": val, "test": test}, out_dir=tmp_path
    )
    assert manifest["splits"]["train"]["rows"] == len(train)
    saved = json.loads((tmp_path / "split_manifest.json").read_text(encoding="utf-8"))
    assert saved["random_seed"] == 42
    loaded = load_split("validation", directory=tmp_path)
    assert list(loaded["customerID"]) == list(val["customerID"])
    with pytest.raises(ValueError):
        load_split("holdout", directory=tmp_path)