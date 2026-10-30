"""Tests for permutation importance and SHAP explanations."""
import numpy as np
import pytest

from churn.explain.permutation import compute_permutation_importance
from churn.explain.plots import (
    plot_bar,
    plot_local_explanation,
    plot_shap_summary,
    plot_signed_bar,
)
from churn.explain.shap_explainer import (
    ChurnExplainer,
    explain_customer,
    feature_groups,
    global_feature_table,
    global_group_table,
    model_specific_importance,
    pick_example_customers,
)
from churn.features.splitting import split_xy, stratified_split
from churn.models.calibration import build_calibrated
from churn.models.pipeline import build_model_pipeline


@pytest.fixture
def data(synthetic_df):
    train, val, _ = stratified_split(synthetic_df)
    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    return X_train, y_train, X_val, y_val


@pytest.fixture
def lr(data):
    X_train, y_train, _, _ = data
    return build_model_pipeline("logistic_regression", "none", seed=42).fit(X_train, y_train)


@pytest.fixture
def lr_explainer(lr, data):
    return ChurnExplainer(lr, data[0], max_background=100)


def test_feature_groups():
    names = ["tenure", "addon_count", "SeniorCitizen", "Contract_Month-to-month",
             "tenure_group_0-12", "PaymentMethod_Electronic check", "gender_Male"]
    assert feature_groups(names) == ["tenure", "addon_count", "SeniorCitizen", "Contract",
                                     "tenure_group", "PaymentMethod", "gender"]


def test_permutation_importance_covers_every_raw_column(lr, data):
    _, _, X_val, y_val = data
    table = compute_permutation_importance(lr, X_val, y_val, n_repeats=3)
    assert set(table["feature"]) == set(X_val.columns)
    assert table["importance_mean"].is_monotonic_decreasing
    assert table["share_positive"].between(0, 1).all()


def test_linear_explainer_is_additive(lr_explainer, data):
    X_val = data[2]
    assert lr_explainer.kind == "linear" and lr_explainer.units == "log-odds"
    shap_values = lr_explainer.shap_values(X_val)
    assert shap_values.shape == (len(X_val), len(lr_explainer.feature_names))
    total = lr_explainer.expected_value + shap_values.sum(axis=1)
    np.testing.assert_allclose(total, lr_explainer.model_output(X_val), atol=1e-5)


def test_tree_explainer_is_additive(data):
    X_train, y_train, X_val, _ = data
    gb = build_model_pipeline("gradient_boosting", "none", seed=42)
    gb.set_params(model__n_estimators=20).fit(X_train, y_train)
    explainer = ChurnExplainer(gb, X_train, max_background=50)
    assert explainer.kind == "tree"
    total = explainer.expected_value + explainer.shap_values(X_val).sum(axis=1)
    np.testing.assert_allclose(total, explainer.model_output(X_val), atol=1e-4)


def test_calibrated_estimator_is_explained_through_its_pipeline(data):
    X_train, y_train, X_val, _ = data
    calibrated = build_calibrated(build_model_pipeline("logistic_regression", "none", seed=42),
                                  "sigmoid", n_splits=3).fit(X_train, y_train)
    explainer = ChurnExplainer(calibrated, X_train, max_background=50)
    assert explainer.kind == "linear"
    assert explainer.shap_values(X_val).shape[1] == len(explainer.feature_names)


def test_global_tables(lr_explainer, data):
    X_val = data[2]
    shap_values = lr_explainer.shap_values(X_val)
    X_t = lr_explainer.transform(X_val)
    features = global_feature_table(shap_values, X_t, lr_explainer.feature_names,
                                    lr_explainer.groups)
    assert len(features) == len(lr_explainer.feature_names)
    groups = global_group_table(shap_values, lr_explainer.groups)
    assert groups["share_pct"].sum() == pytest.approx(100.0)
    assert "Contract" in set(groups["group"])


def test_direction_of_effect_on_known_values():
    shap_values = np.array([[1.0], [1.0], [-1.0], [-1.0]])
    X_t = np.array([[1.0], [1.0], [0.0], [0.0]])
    table = global_feature_table(shap_values, X_t, ["flag"], ["flag"])
    assert table.loc[0, "direction"] == "raises risk when present"


def test_explain_customer_adds_up(lr_explainer, data):
    row = data[2].iloc[[0]]
    result = explain_customer(lr_explainer, row, top_n=5)
    contributions = result["contributions"]
    assert len(contributions) == 6 and contributions.iloc[-1]["group"] == "other features"
    assert result["base_value"] + contributions["shap"].sum() == pytest.approx(
        result["model_output"], abs=1e-5)
    np.testing.assert_allclose(contributions["odds_multiplier"], np.exp(contributions["shap"]))
    with pytest.raises(ValueError):
        explain_customer(lr_explainer, data[2].iloc[:2])


def test_model_specific_importance_for_logistic_regression(lr_explainer):
    table = model_specific_importance(lr_explainer)
    assert len(table) == len(lr_explainer.feature_names)
    np.testing.assert_allclose(table["odds_ratio"], np.exp(table["coefficient"]))


def test_pick_example_customers():
    examples = pick_example_customers([0.1, 0.9, 0.5, 0.45, 0.2], threshold=0.4)
    assert examples == {"highest_risk": 1, "typical_flagged": 2,
                        "borderline_flagged": 3, "lowest_risk": 0}


def test_plots(lr_explainer, data, tmp_path):
    X_val = data[2]
    shap_values = lr_explainer.shap_values(X_val)
    X_t = lr_explainer.transform(X_val)
    local = explain_customer(lr_explainer, X_val.iloc[[0]])["contributions"]
    paths = [
        plot_bar(["a", "b"], [0.2, 0.1], tmp_path / "bar.png", "t", "x", errors=[0.01, 0.02]),
        plot_signed_bar(["a", "b"], [0.2, -0.1], tmp_path / "signed.png", "t", "x"),
        plot_shap_summary(shap_values, X_t, lr_explainer.feature_names, tmp_path / "summary.png"),
        plot_local_explanation(local, tmp_path / "local.png", "customer"),
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)