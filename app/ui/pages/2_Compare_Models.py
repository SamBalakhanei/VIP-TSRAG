import pandas as pd
import streamlit as st

from api_client import get_leaderboard, list_models

st.title("Compare Models")

models = list_models()
model_names = [m["model_name"] for m in models]
selected_models = st.multiselect("Select models", model_names, default=model_names[:2])
metric_name = st.selectbox("Metric", ["MASE", "MAE", "RMSE"], index=0)

rows = get_leaderboard(metric_name=metric_name)
df = pd.DataFrame(rows)

if df.empty:
    st.warning("No comparison data found.")
else:
    filtered = df[df["model_name"].isin(selected_models)]
    st.dataframe(filtered, use_container_width=True)
    if not filtered.empty:
        st.bar_chart(filtered.set_index("model_name")["mean_score"])