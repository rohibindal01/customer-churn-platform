"""Exploratory data analysis helpers for the Telco churn dataset.

Functions here are pure and reusable: they take a DataFrame and return
tables or save figures. scripts/02_run_eda.py is the thin entry point.
Every number in the report is computed from the data at run time.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files only, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

from churn.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
CHURN_FLAG = "churn_flag"
STATUS = "churn_status"
TENURE_GROUP = "tenure_group"
PALETTE = {"Retained": "#4C72B0", "Churned": "#C44E52"}

_TENURE_BINS = [-1, 12, 24, 48, 60, np.inf]
_TENURE_LABELS = ["0-12", "13-24", "25-48", "49-60", "61+"]


# ---------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------
def prepare_eda_frame(df: pd.DataFrame, target: str, positive_label: str) -> pd.DataFrame:
    """Return a copy with TotalCharges as numbers plus helper columns.

    Blank TotalCharges strings become NaN. Helper columns added:
    churn_flag (1/0), churn_status (text) and tenure_group (ordered bands).
    """
    required = {target, "tenure", "MonthlyCharges", "TotalCharges"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    cleaned = out["TotalCharges"].astype("string").str.strip()
    out["TotalCharges"] = pd.to_numeric(cleaned, errors="coerce")

    is_positive = out[target].astype("string").str.strip() == positive_label
    out[CHURN_FLAG] = is_positive.fillna(False).astype(int)
    out[STATUS] = np.where(out[CHURN_FLAG] == 1, "Churned", "Retained")
    out[TENURE_GROUP] = pd.cut(out["tenure"], bins=_TENURE_BINS, labels=_TENURE_LABELS)
    return out


def categorical_columns(df: pd.DataFrame, id_column: str, target: str) -> list[str]:
    """Columns analysed as categories (SeniorCitizen is a 0/1 category)."""
    skip = {id_column, target, CHURN_FLAG, STATUS, TENURE_GROUP, *NUMERIC_COLUMNS}
    return [c for c in df.columns if c not in skip]


def _display_labels(df: pd.DataFrame, column: str) -> pd.Series:
    if column == "SeniorCitizen":
        return df[column].map({0: "No", 1: "Yes"}).rename(column)
    return df[column]


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------
def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Dtype, null count and unique count for every column."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "nulls": df.isna().sum(),
            "unique": df.nunique(),
        }
    ).rename_axis("column").reset_index()


def iqr_outlier_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Count values outside 1.5 * IQR for each numeric column."""
    rows = []
    for col in columns:
        values = df[col].dropna()
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "lower_fence": lower,
                "upper_fence": upper,
                "outliers": n_out,
                "outlier_pct": round(100 * n_out / len(values), 2) if len(values) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def churn_rate_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Customers, churners and churn rate for each category of one column."""
    labels = _display_labels(df, column)
    grouped = df.groupby(labels, observed=True)[CHURN_FLAG].agg(
        customers="size", churned="sum"
    )
    grouped["churn_rate_pct"] = (grouped["churned"] / grouped["customers"] * 100).round(2)
    out = grouped.reset_index().rename(columns={column: "category"})
    out["category"] = out["category"].astype(str)
    out.insert(0, "feature", column)
    return out


def segments_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Churn rate for every category of every column, stacked in one table."""
    names = list(columns)
    if TENURE_GROUP in df.columns:
        names.append(TENURE_GROUP)
    return pd.concat([churn_rate_table(df, c) for c in names], ignore_index=True)


def categorical_association_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Chi-square test and Cramer's V of each categorical column against churn."""
    rows = []
    for col in columns:
        table = pd.crosstab(df[col], df[CHURN_FLAG])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p_value, dof, _ = stats.chi2_contingency(table, correction=False)
        n = table.to_numpy().sum()
        k = min(table.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else float("nan")
        rows.append(
            {
                "feature": col,
                "categories": int(table.shape[0]),
                "chi2": float(chi2),
                "dof": int(dof),
                "p_value": float(p_value),
                "cramers_v": cramers_v,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("cramers_v", ascending=False).reset_index(drop=True)


def numeric_association_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Mann-Whitney U test and rank-biserial effect, churned vs retained.

    rank_biserial is between -1 and 1. Positive means churned customers tend
    to have larger values than retained customers.
    """
    rows = []
    for col in columns:
        valid = df[[col, CHURN_FLAG]].dropna()
        churned = valid.loc[valid[CHURN_FLAG] == 1, col]
        retained = valid.loc[valid[CHURN_FLAG] == 0, col]
        if len(churned) == 0 or len(retained) == 0:
            continue
        result = stats.mannwhitneyu(churned, retained, alternative="two-sided")
        rank_biserial = 2 * float(result.statistic) / (len(churned) * len(retained)) - 1
        rows.append(
            {
                "feature": col,
                "median_churned": float(churned.median()),
                "median_retained": float(retained.median()),
                "mean_churned": float(churned.mean()),
                "mean_retained": float(retained.mean()),
                "p_value": float(result.pvalue),
                "rank_biserial": rank_biserial,
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of the numeric columns, SeniorCitizen and churn."""
    cols = [c for c in NUMERIC_COLUMNS + ["SeniorCitizen", CHURN_FLAG] if c in df.columns]
    return df[cols].corr(method="spearman")


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_churn_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = df[STATUS].value_counts().reindex(["Retained", "Churned"]).fillna(0)
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[k] for k in counts.index])
    for bar, value in zip(bars, counts.values):
        pct = 100 * value / total if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value)}\n({pct:.1f}%)",
                ha="center", va="bottom")
    ax.set_ylabel("Customers")
    ax.set_title("Churn distribution")
    ax.set_ylim(0, counts.max() * 1.18 if counts.max() > 0 else 1)
    return _save(fig, out_dir / "churn_distribution.png")


def plot_numeric_distributions(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    fig, axes = plt.subplots(2, len(cols), figsize=(6 * len(cols), 9), squeeze=False)
    for i, col in enumerate(cols):
        sns.histplot(data=df, x=col, hue=STATUS, palette=PALETTE, stat="density",
                     common_norm=False, element="step", bins=30, ax=axes[0, i])
        axes[0, i].set_title(f"{col} distribution by churn status")
        sns.boxplot(data=df, x=STATUS, y=col, hue=STATUS, palette=PALETTE,
                    order=["Retained", "Churned"], hue_order=["Retained", "Churned"],
                    legend=False, ax=axes[1, i])
        axes[1, i].set_title(f"{col} by churn status")
        axes[1, i].set_xlabel("")
    return _save(fig, out_dir / "numeric_distributions.png")


def plot_categorical_churn_rates(df: pd.DataFrame, columns: list[str], out_dir: Path) -> Path:
    overall = df[CHURN_FLAG].mean() * 100
    n_cols = 4
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows), squeeze=False)
    flat = axes.ravel()
    for ax, col in zip(flat, columns):
        table = churn_rate_table(df, col).sort_values("churn_rate_pct")
        labels = [textwrap.fill(str(s), 22) for s in table["category"]]
        ax.barh(labels, table["churn_rate_pct"], color="#4C72B0")
        ax.axvline(overall, color="#C44E52", linestyle="--", linewidth=1)
        ax.set_title(col)
        ax.set_xlabel("Churn rate (%)")
    for ax in flat[len(columns):]:
        ax.set_visible(False)
    fig.suptitle(f"Churn rate by category (red line = overall {overall:.1f}%)", y=1.0)
    return _save(fig, out_dir / "categorical_churn_rates.png")


def plot_service_churn_rates(df: pd.DataFrame, out_dir: Path) -> Path | None:
    """Churn rate with and without each add-on service, internet customers only."""
    if "InternetService" not in df.columns:
        return None
    internet = df[df["InternetService"] != "No"]
    rows = []
    for service in SERVICE_COLUMNS:
        if service not in internet.columns:
            continue
        for value in ("Yes", "No"):
            subset = internet[internet[service] == value]
            if len(subset):
                rows.append({"service": service, "has_service": value,
                             "churn_rate_pct": subset[CHURN_FLAG].mean() * 100})
    data = pd.DataFrame(rows)
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=data, x="service", y="churn_rate_pct", hue="has_service",
                hue_order=["Yes", "No"], ax=ax)
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("")
    ax.set_title("Churn rate with vs without each add-on (customers with internet service)")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, out_dir / "services_churn_rates.png")


def plot_tenure_group_churn(df: pd.DataFrame, out_dir: Path) -> Path:
    table = churn_rate_table(df, TENURE_GROUP)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(table["category"], table["churn_rate_pct"], color="#4C72B0")
    for bar, n in zip(bars, table["customers"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={int(n)}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Tenure group (months)")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rate by tenure group")
    return _save(fig, out_dir / "tenure_group_churn.png")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    corr = correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Spearman correlation (numeric features and churn)")
    return _save(fig, out_dir / "correlation_heatmap.png")


def write_contract_internet_heatmap(df: pd.DataFrame, out_dir: Path) -> Path | None:
    if not {"Contract", "InternetService"} <= set(df.columns):
        return None
    pivot = df.pivot_table(index="Contract", columns="InternetService",
                           values=CHURN_FLAG, aggfunc="mean") * 100
    fig = px.imshow(pivot.round(1), text_auto=True, aspect="auto",
                    color_continuous_scale="Reds", labels={"color": "Churn rate (%)"},
                    title="Churn rate (%) by contract type and internet service")
    path = out_dir / "contract_internet_heatmap.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def write_tenure_charges_scatter(df: pd.DataFrame, out_dir: Path) -> Path:
    hover = [c for c in ("Contract", "InternetService") if c in df.columns]
    fig = px.scatter(df, x="tenure", y="MonthlyCharges", color=STATUS,
                     color_discrete_map=PALETTE, opacity=0.6, hover_data=hover,
                     title="Tenure vs monthly charges, coloured by churn status")
    path = out_dir / "tenure_vs_monthly_scatter.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


# ---------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------
def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Cannot serialise {type(obj)}")


def _fmt(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.3g}"
    return str(value)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no rows)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _build_report(summary: dict, cat_assoc: pd.DataFrame, num_assoc: pd.DataFrame,
                  outliers: pd.DataFrame, segments: pd.DataFrame, min_segment_size: int) -> str:
    eligible = segments[segments["customers"] >= min_segment_size]
    highest = eligible.sort_values("churn_rate_pct", ascending=False).head(8)
    lowest = eligible.sort_values("churn_rate_pct", ascending=True).head(8)
    seg_cols = ["feature", "category", "customers", "churned", "churn_rate_pct"]

    blank_note = ""
    if summary["blank_total_charges"]:
        blank_note = (
            f"{summary['blank_total_charges']} TotalCharges values are blank. "
            f"All of them have tenure == 0: {summary['blank_total_charges_all_zero_tenure']}."
        )

    parts = [
        "# EDA report",
        "",
        f"Source file: {summary['source_file']} (fingerprint {summary['fingerprint']}). "
        "Generated by scripts/02_run_eda.py. All numbers are computed from the data.",
        "",
        "## 1. Dataset snapshot",
        f"- Rows: {summary['rows']}, columns: {summary['columns']}",
        f"- Duplicate rows: {summary['duplicate_rows']}, duplicated customer IDs: {summary['duplicate_ids']}",
        f"- {blank_note or 'No blank TotalCharges values.'}",
        "",
        "## 2. Class balance",
        f"- Churned: {summary['churned']} ({summary['churn_rate_pct']:.2f}%), "
        f"retained: {summary['retained']}",
        "",
        "## 3. Strongest categorical associations with churn (Cramer's V)",
        "Rough guide: below 0.1 weak, 0.1 to 0.3 moderate, above 0.3 strong. "
        "Association is not causation.",
        "",
        _md_table(cat_assoc.head(10)[["feature", "categories", "cramers_v", "p_value"]]),
        "",
        "## 4. Numeric features, churned vs retained (Mann-Whitney U)",
        "rank_biserial above 0 means churned customers tend to have larger values.",
        "",
        _md_table(num_assoc),
        "",
        "## 5. Outliers (1.5 x IQR rule)",
        "Reported, not removed. High charges are real customers, not errors.",
        "",
        _md_table(outliers),
        "",
        f"## 6. Highest-churn segments (at least {min_segment_size} customers)",
        _md_table(highest[seg_cols]),
        "",
        f"## 7. Lowest-churn segments (at least {min_segment_size} customers)",
        _md_table(lowest[seg_cols]),
        "",
        "## 8. Caveats",
        "- Segments overlap: one customer can appear in several rows above.",
        "- Features are correlated with each other (for example tenure and TotalCharges), "
        "so a segment's churn rate is not the effect of that one feature alone.",
        "- These are historical associations in one dataset. They do not prove that "
        "changing a feature would change churn.",
        "",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
def run_eda(
    df: pd.DataFrame,
    output_dir: Path | str,
    target: str = "Churn",
    positive_label: str = "Yes",
    id_column: str = "customerID",
    source_name: str = "in-memory",
    fingerprint: str = "n/a",
    min_segment_size: int = 100,
) -> dict:
    """Run the full EDA, write tables, figures, JSON summary and Markdown report."""
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    tab_dir = output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_eda_frame(df, target, positive_label)
    cat_cols = categorical_columns(data, id_column, target)
    logger.info("EDA started: %d rows, %d categorical columns", len(data), len(cat_cols))

    overview = dataset_overview(df)
    outliers = iqr_outlier_table(data, NUMERIC_COLUMNS)
    cat_assoc = categorical_association_table(data, cat_cols)
    num_assoc = numeric_association_table(data, NUMERIC_COLUMNS)
    segments = segments_table(data, cat_cols)
    corr = correlation_matrix(data)

    tables = {
        "dataset_overview": overview,
        "outliers_iqr": outliers,
        "categorical_association": cat_assoc,
        "numeric_association": num_assoc,
        "churn_rate_by_segment": segments,
        "spearman_correlation": corr.reset_index().rename(columns={"index": "feature"}),
    }
    files: list[Path] = []
    for name, table in tables.items():
        path = tab_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        files.append(path)

    logger.info("Saving figures")
    figure_paths = [
        plot_churn_distribution(data, fig_dir),
        plot_numeric_distributions(data, fig_dir),
        plot_categorical_churn_rates(data, cat_cols, fig_dir),
        plot_service_churn_rates(data, fig_dir),
        plot_tenure_group_churn(data, fig_dir),
        plot_correlation_heatmap(data, fig_dir),
        write_contract_internet_heatmap(data, fig_dir),
        write_tenure_charges_scatter(data, fig_dir),
    ]
    files.extend(p for p in figure_paths if p is not None)

    blank_mask = data["TotalCharges"].isna()
    blank_count = int(blank_mask.sum())
    summary = {
        "source_file": source_name,
        "fingerprint": fingerprint,
        "rows": int(len(data)),
        "columns": int(df.shape[1]),
        "churned": int(data[CHURN_FLAG].sum()),
        "retained": int((data[CHURN_FLAG] == 0).sum()),
        "churn_rate_pct": float(data[CHURN_FLAG].mean() * 100),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[id_column].duplicated().sum()) if id_column in df.columns else 0,
        "blank_total_charges": blank_count,
        "blank_total_charges_all_zero_tenure": (
            bool((data.loc[blank_mask, "tenure"] == 0).all()) if blank_count else None
        ),
        "top_categorical_drivers": (
            cat_assoc.head(5)[["feature", "cramers_v"]].to_dict("records")
            if not cat_assoc.empty else []
        ),
        "numeric_tests": num_assoc.to_dict("records"),
    }

    summary_path = output_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    report_path = output_dir / "eda_report.md"
    report_path.write_text(
        _build_report(summary, cat_assoc, num_assoc, outliers, segments, min_segment_size),
        encoding="utf-8",
    )
    files.extend([summary_path, report_path])

    logger.info("EDA finished: %d files written to %s", len(files), output_dir)
    return {"summary": summary, "files": files}