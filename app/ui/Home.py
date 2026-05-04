import pandas as pd
import streamlit as st

from api_client import get_leaderboard

st.set_page_config(page_title="VIP Benchmark Dashboard", layout="wide")

st.title("VIP Benchmark Dashboard")
st.write("Home page contains a table of all the metric results of our three models. Navigate to leaderboard page to view a full benchmark leaderboard you can filter based on model, metric, horizon, etc.")

metrics = ["MAE", "MSE", "MASE", "WQL"]

all_rows = []

for metric in metrics:
    try:
        rows = get_leaderboard(metric_name=metric)
        all_rows.extend(rows)
    except Exception:
        pass

df = pd.DataFrame(all_rows)

if df.empty:
    st.warning("No benchmark data found.")
else:
    display_df = df[
        ["model_name", "family", "metric_name", "mean_score"]
    ].copy()

    display_df = display_df.rename(columns={
        "model_name": "Model Name",
        "family": "Model Family",
        "metric_name": "Metric Name",
        "mean_score": "Mean Score",
    })

    display_df["Mean Score"] = display_df["Mean Score"].round(6)

    st.subheader("Full Metrics Table")
    display_df = display_df.pivot_table(
        index=["Model Name", "Model Family"],
        columns="Metric Name",
        values="Mean Score"
    ).reset_index()

    st.dataframe(display_df, use_container_width=True, hide_index=True)



