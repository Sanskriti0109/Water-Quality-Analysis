import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from loader import load_and_clean, get_outliers
from ml_models import train_models
from charts import (
    plot_correlation_heatmap, plot_polar, plot_time_series,
    plot_3d_surface, plot_network_graph,
    plot_model_comparison, plot_feature_importance
)
# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Water Quality Analysis",
    page_icon="💧",
    layout="wide"
)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("💧 Water Quality Analysis")
st.sidebar.markdown("**PBL 2026 | Sanskriti Saxena**")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🔍 EDA",
    "🤖 Model Results",
    "🔮 Predict DO"
])

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def get_data(file):
    return load_and_clean(file)

@st.cache_data
def get_models(file):
    df = load_and_clean(file)
    return train_models(df)

if uploaded_file:
    df = get_data(uploaded_file)
else:
    st.info("👈 Upload your `brisbane_water_quality.csv` from the sidebar to get started.")
    st.stop()

# ── Page: Overview ────────────────────────────────────────────
if page == "📊 Overview":
    st.title("Water Quality Analysis Dashboard")
    st.markdown("Brisbane Environmental Monitoring | 30-minute interval sensor data")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Features", len(df.columns) - 1)
    col3.metric("Avg DO (mg/L)", f"{df['Dissolved Oxygen'].mean():.2f}")
    col4.metric("Avg Temperature (°C)", f"{df['Temperature'].mean():.2f}")

    st.subheader("Raw Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Outlier Summary")
    outliers = get_outliers(df)
    st.warning(f"{len(outliers)} outlier rows detected via Z-score (> 3 std dev)")
    with st.expander("View outlier rows"):
        st.dataframe(outliers, use_container_width=True)

# ── Page: EDA ────────────────────────────────────────────────
elif page == "🔍 EDA":
    st.title("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Correlation Heatmap", "Polar Plot",
        "Time Series", "3D Surface", "Network Graph"
    ])

    with tab1:
        st.pyplot(plot_correlation_heatmap(df))

    with tab2:
        st.pyplot(plot_polar(df))

    with tab3:
        st.pyplot(plot_time_series(df))

    with tab4:
        st.pyplot(plot_3d_surface(df))

    with tab5:
        st.pyplot(plot_network_graph(df))

# ── Page: Model Results ───────────────────────────────────────
elif page == "🤖 Model Results":
    st.title("Machine Learning Model Results")

    with st.spinner("Training models... this may take a moment ⏳"):
        results, y_test, X, scaler = get_models(uploaded_file)

    col1, col2, col3 = st.columns(3)
    for col, name in zip([col1, col2, col3], results.keys()):
        col.metric(f"{name} R²", f"{results[name]['r2']:.5f}")
        col.metric(f"{name} RMSE", f"{results[name]['rmse']:.5f} mg/L")

    st.subheader("Model Comparison")
    st.pyplot(plot_model_comparison(results))

    st.subheader("Feature Importance (Random Forest)")
    st.pyplot(plot_feature_importance(results))

# ── Page: Predict DO ─────────────────────────────────────────
elif page == "🔮 Predict DO":
    st.title("Predict Dissolved Oxygen")
    st.markdown("Adjust the sliders to predict DO level using the **Random Forest** model.")

    with st.spinner("Loading model..."):
        results, _, X, scaler = get_models(uploaded_file)

    rf_model = results['Random Forest']['model']
    feature_names = results['Random Forest']['feature_names']

    input_data = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        col = cols[i % 3]
        min_val = float(X[feat].min())
        max_val = float(X[feat].max())
        mean_val = float(X[feat].mean())
        input_data[feat] = col.slider(feat, min_val, max_val, mean_val)

    input_df = pd.DataFrame([input_data])

    if st.button("🔮 Predict", type="primary"):
        prediction = rf_model.predict(input_df)[0]
        st.success(f"### Predicted Dissolved Oxygen: **{prediction:.3f} mg/L**")
        if prediction < 5:
            st.error("⚠️ Warning: DO below 5 mg/L — hypoxic risk for aquatic life!")
        elif prediction < 7:
            st.warning("🟡 Moderate DO level")
        else:
            st.success("✅ Healthy DO level")