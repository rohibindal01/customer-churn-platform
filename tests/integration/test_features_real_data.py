"""Integration tests: split and feature pipeline on the real dataset."""
import numpy as np
import pytest

from churn.config.settings import get_settings
from churn.data.loader import load_raw_data
from churn.features.cleaning import drop_duplicate_rows
from churn.features.preprocessing import build_feature_pipeline, get_feature_names
from churn.features.splitting import make_target, split_xy, stratified_split

pytestmark = pytest.mark.skipif(
    not get_settings().raw_data_path.exists(),
    reason="Raw dataset not downloaded yet (run scripts\\01_download_data.py)",
)


def test_real_split_is_disjoint_and_stratified():
    df, _ = drop_duplicate_rows(load_raw_data())
    train, val, test = stratified_split(df)
    ids = [set(f["customerID"]) for f in (train, val, test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert sum(len(f) for f in (train, val, test)) == len(df)
    overall = make_target(df).mean()
    for frame in (train, val, test):
        assert make_target(frame).mean() == pytest.approx(overall, abs=0.01)


def test_real_feature_pipeline_gives_clean_matrices():
    df, _ = drop_duplicate_rows(load_raw_data())
    train, val, test = stratified_split(df)
    X_train, _ = split_xy(train)
    pipe = build_feature_pipeline()
    train_out = pipe.fit_transform(X_train)
    width = len(get_feature_names(pipe))
    assert train_out.shape[1] == width
    for frame in (val, test):
        X, _ = split_xy(frame)
        out = pipe.transform(X)
        assert out.shape[1] == width
        assert not np.isnan(out).any()
    assert not np.isnan(train_out).any()