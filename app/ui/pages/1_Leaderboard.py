import pandas as pd
import streamlit as st

from api_client import get_leaderboard

st.title("Leaderboard")
st.caption("TS-RAG vs RAF vs GTR")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    metric_name = st.selectbox("Metric", ["MAE", "MASE", "RMSE"], index=1)

with col2:
    family = st.selectbox("Model Family", ["All", "RAF", "GTR", "TS-RAG"], index=0)
    family = None if family == "All" else family

with col3:
    sector = st.selectbox("Sector", ["All", "Tech", "Finance", "Healthcare"], index=0)
    sector = None if sector == "All" else sector

with col4:
    horizon_choice = st.selectbox("Horizon", ["All", 16, 32], index=0)
    horizon = None if horizon_choice == "All" else int(horizon_choice)

with col5:
    lookback_choice = st.selectbox("Lookback", ["All", 64, 128], index=0)
    lookback = None if lookback_choice == "All" else int(lookback_choice)


rows = get_leaderboard(
    metric_name=metric_name,
    family=family,
    sector=sector,
    horizon=horizon,
    lookback=lookback,
)

df = pd.DataFrame(rows)

if df.empty:
    st.warning("No leaderboard rows found for the selected filters.")
    st.stop()


best_row = df.iloc[0]
num_models = df["model_name"].nunique()
num_tasks = df["task_name"].nunique()
avg_score = df["mean_score"].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Best Model", best_row["model_name"])
m2.metric("Best Score", f"{best_row['mean_score']:.3f}")
m3.metric("Models Shown", num_models)
m4.metric("Avg Score", f"{avg_score:.3f}")

st.divider()


display_df = df[
    [
        "rank",
        "model_name",
        "family",
        "task_name",
        "sector",
        "horizon",
        "lookback",
        "metric_name",
        "mean_score",
        "num_runs",
        "num_series",
    ]
].copy()

display_df["mean_score"] = display_df["mean_score"].round(4)

st.subheader("Filtered Leaderboard")
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

st.divider()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Model Performance")
    model_chart = (
        df.groupby("model_name", as_index=False)["mean_score"]
        .mean()
        .sort_values("mean_score", ascending=True)
    )
    st.bar_chart(model_chart.set_index("model_name")["mean_score"])

with chart_col2:
    st.subheader("Sector Performance")
    sector_chart = (
        df.groupby(["sector", "model_name"], as_index=False)["mean_score"]
        .mean()
        .sort_values(["sector", "mean_score"])
    )
    pivot_sector = sector_chart.pivot(index="sector", columns="model_name", values="mean_score")
    st.bar_chart(pivot_sector)

st.divider()

bottom_col1, bottom_col2 = st.columns(2)

with bottom_col1:
    st.subheader("Performance by Horizon")
    horizon_chart = (
        df.groupby(["horizon", "model_name"], as_index=False)["mean_score"]
        .mean()
        .sort_values(["horizon", "mean_score"])
    )
    pivot_horizon = horizon_chart.pivot(index="horizon", columns="model_name", values="mean_score")
    st.line_chart(pivot_horizon)

with bottom_col2:
    st.subheader("Raw Data Preview")
    st.dataframe(df, use_container_width=True, hide_index=True)