"""Batch Prediction: validate and score a CSV file of customers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app\components importable

import io  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from churn.serving.predictor import MAX_BATCH_ROWS, BatchInputError  # noqa: E402
from components.loaders import artifact_bytes, get_service  # noqa: E402
from components.ui import (  # noqa: E402
    BAND_COLORS,
    BAND_ORDER,
    disclaimer,
    page_header,
    require_service,
    setup_page,
    show_chart,
    show_df,
    sidebar_model_card,
    threshold_control,
)

UPLOAD = "Upload a CSV file"
SAMPLES = {
    "Sample: 25 valid customers": "artifacts/serving/sample_batch.csv",
    "Sample: 5 rows, 3 deliberately invalid": "artifacts/serving/sample_batch_with_errors.csv",
}

setup_page("Batch Prediction", "📂")
service = require_service()
threshold = threshold_control(service)
sidebar_model_card(service)
page_header("Batch Prediction",
            "Upload a CSV of customers, or try a built-in sample. Every row is checked: valid "
            "rows are scored, invalid rows are listed with the reason.")

with st.expander("File format and samples"):
    st.markdown("**Required columns:** " + ", ".join(f"`{c}`" for c in service.input_columns))
    st.markdown("**Optional:** `customerID` (kept in the output). A `Churn` column is ignored. "
                f"Up to {MAX_BATCH_ROWS:,} rows. A blank `TotalCharges` is allowed for new customers.")
    left, right = st.columns(2)
    sample = artifact_bytes(SAMPLES["Sample: 25 valid customers"])
    broken = artifact_bytes(SAMPLES["Sample: 5 rows, 3 deliberately invalid"])
    if sample:
        left.download_button("Download a sample file", sample, "sample_batch.csv", "text/csv")
    if broken:
        right.download_button("Download a sample with invalid rows", broken,
                              "sample_batch_with_errors.csv", "text/csv")

# Built-in samples go through exactly the same code path as an uploaded file.
sources = [UPLOAD, *[label for label, rel in SAMPLES.items() if artifact_bytes(rel) is not None]]
source = st.radio("Data source", sources, horizontal=True, key="batch_source",
                  help="The samples are held-out validation customers, so you can try the page "
                       "without preparing a file.")
explain = st.checkbox("Add each customer's top risk driver and suggested actions", value=True,
                      key="batch_explain", help="Slower for very large files.")

if source == UPLOAD:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], key="batch_upload")
    if uploaded is None:
        st.info("Upload a file to start, or choose a sample in Data source.")
        st.stop()
    content, file_name = uploaded.getvalue(), uploaded.name
else:
    content, file_name = artifact_bytes(SAMPLES[source]), Path(SAMPLES[source]).name


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    """Read every column as text, so blanks and typos can be reported per row."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False,
                               encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("unsupported text encoding")


@st.cache_data(show_spinner=False, max_entries=5)
def score(content: bytes, threshold: float, explain: bool) -> pd.DataFrame:
    return get_service().predict_batch(read_csv_bytes(content), threshold=threshold,
                                       explain=explain)


try:
    frame = read_csv_bytes(content)
except Exception as exc:  # noqa: BLE001 - show any parsing problem to the user
    st.error(f"The file could not be read as CSV: {exc}")
    st.stop()
st.caption(f"Read {len(frame):,} rows and {frame.shape[1]} columns from {file_name}.")

problems = service.check_file(frame)
if problems:
    for problem in problems:
        st.error(problem)
    st.stop()
if "customerID" in frame.columns:
    duplicates = int(frame["customerID"].duplicated().sum())
    if duplicates:
        st.warning(f"{duplicates} duplicated customerID value(s). Each row is scored independently.")

try:
    with st.spinner("Validating and scoring..."):
        results = score(content, threshold, explain)
except BatchInputError as exc:
    for message in exc.errors:
        st.error(message)
    st.stop()

scored = results[results["status"] == "scored"]
rejected = results[results["status"] == "rejected"]
cols = st.columns(4)
cols[0].metric("Rows", f"{len(results):,}")
cols[1].metric("Scored", f"{len(scored):,}")
cols[2].metric("Rejected", f"{len(rejected):,}")
cols[3].metric("Predicted to churn", f"{(scored['predicted_churn'] == 'Yes').sum():,}")

if len(rejected):
    st.warning(f"{len(rejected)} row(s) could not be scored. They stay in the download with "
               "the reason, so you can fix them.")
    with st.expander("Rejected rows"):
        show_df(rejected[[c for c in ("row", "customerID", "problem") if c in rejected.columns]],
                hide_index=True)

if len(scored):
    left, right = st.columns(2)
    with left:
        counts = scored["risk_band"].value_counts().reindex(BAND_ORDER, fill_value=0)
        fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                     color_discrete_map=BAND_COLORS, text=counts.values,
                     labels={"x": "Risk band", "y": "Customers"}, title="Risk bands")
        fig.update_layout(showlegend=False, height=340)
        show_chart(fig)
    with right:
        fig = px.histogram(scored, x="churn_probability", nbins=20,
                           labels={"churn_probability": "Churn probability"},
                           title="Distribution of churn probabilities")
        fig.add_vline(x=threshold, line_dash="dash", line_color="#C0392B")
        fig.update_layout(height=340)
        show_chart(fig)

st.subheader("Results")
show_df(results, hide_index=True)
st.download_button("Download results (CSV)", results.to_csv(index=False).encode("utf-8"),
                   file_name=f"churn_scores_{Path(file_name).stem}.csv", mime="text/csv",
                   type="primary")
disclaimer()