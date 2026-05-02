"""
app.py — AutoML Document Analyzer (Streamlit UI)
─────────────────────────────────────────────────
Run with:
    streamlit run app.py
"""

import io
import json

import numpy as np
import pandas as pd
import streamlit as st

import eda
import model as mdl
import preprocess as prep

# ─────────────────────────────────────────────────────────────────────────────
# Page config & global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Document Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0D0D1A;
    --surface:   #12122A;
    --card:      #1A1A35;
    --border:    #2E2E55;
    --accent1:   #4F8EF7;
    --accent2:   #F7874F;
    --accent3:   #4FF7A8;
    --text:      #E0E0FF;
    --muted:     #9090B0;
    --font-main: 'Space Grotesk', sans-serif;
    --font-mono: 'Fira Code', monospace;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-main) !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: var(--font-main) !important; }

/* ── Hero title ── */
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent1) 0%, var(--accent3) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    line-height: 1.1;
}
.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 4px;
    font-weight: 400;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--accent1);
    border-left: 3px solid var(--accent1);
    padding-left: 10px;
    margin: 1.5rem 0 0.75rem;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent3);
    font-family: var(--font-mono);
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-blue  { background: rgba(79,142,247,.15); color: var(--accent1); border: 1px solid var(--accent1); }
.badge-green { background: rgba(79,247,168,.15); color: var(--accent3); border: 1px solid var(--accent3); }
.badge-orange{ background: rgba(247,135,79,.15); color: var(--accent2); border: 1px solid var(--accent2); }

/* ── Best model banner ── */
.best-model-banner {
    background: linear-gradient(135deg, rgba(79,142,247,.15), rgba(79,247,168,.08));
    border: 1px solid var(--accent1);
    border-radius: 12px;
    padding: 18px 24px;
    margin: 1rem 0;
}
.best-model-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent3);
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent1), #3a6fd8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-main) !important;
    font-weight: 600 !important;
    padding: 8px 24px !important;
}
.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
    transition: all .15s;
}
.stDownloadButton > button {
    background: var(--card) !important;
    color: var(--accent3) !important;
    border: 1px solid var(--accent3) !important;
    border-radius: 8px !important;
    font-family: var(--font-main) !important;
    font-weight: 600 !important;
}
div[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    background: var(--card) !important;
    padding: 12px !important;
}
[data-testid="stDataFrameResizable"] { border-radius: 8px; overflow: hidden; }
.stSelectbox label, .stRadio label { color: var(--text) !important; }

/* ── Table ── */
.stDataFrame { background: var(--card) !important; }

/* step indicator */
.step-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 6px 16px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--muted);
    margin: 4px 4px 4px 0;
}
.step-pill.active { color: var(--accent3); border-color: var(--accent3); }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(value, label: str):
    st.markdown(
        f"""<div class="metric-card">
               <div class="metric-value">{value}</div>
               <div class="metric-label">{label}</div>
             </div>""",
        unsafe_allow_html=True,
    )


def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>',
                unsafe_allow_html=True)


def badge(text: str, color: str = "blue"):
    st.markdown(f'<span class="badge badge-{color}">{text}</span>',
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_file(file_bytes: bytes, file_name: str) -> pd.DataFrame | None:
    """Load CSV or Excel file, return DataFrame or None on error."""
    try:
        if file_name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(file_bytes))
        return pd.read_excel(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"❌ Could not parse file: {e}")
        return None


def clean_results_for_display(results: dict) -> dict:
    """Remove non-serialisable keys (_cm, _y_pred) from results dict."""
    return {
        name: {k: v for k, v in metrics.items() if not k.startswith("_")}
        for name, metrics in results.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """<div style="padding:12px 0 20px">
             <div class="hero-title" style="font-size:1.5rem;">🧠 AutoML</div>
             <div class="hero-sub" style="font-size:.8rem;">Document Analyzer</div>
           </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    pages = [
        "📂 Upload Dataset",
        "🔍 Exploratory Analysis",
        "⚙️  Preprocessing",
        "🤖 Train & Evaluate",
        "📊 Dashboard",
        "💾 Export",
    ]
    page = st.radio("Navigation", pages, label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        "<div style='color:#9090B0;font-size:.72rem;'>AutoML Document Analyzer<br>"
        "Built with Streamlit · scikit-learn</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
for key in ("df", "target", "problem_type", "preprocessor",
            "X_train", "X_test", "y_train", "y_test",
            "trained_models", "results", "best_name",
            "eda_overview"):
    if key not in st.session_state:
        st.session_state[key] = None


# ─────────────────────────────────────────────────────────────────────────────
# Page: Upload
# ─────────────────────────────────────────────────────────────────────────────
if page == pages[0]:
    st.markdown('<div class="hero-title">AutoML Document Analyzer</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Upload a CSV or Excel dataset and let the '
        "pipeline do the rest — EDA, preprocessing, training, evaluation.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your dataset here (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Max 200 MB",
    )

    if uploaded:
        with st.spinner("Loading…"):
            df = load_file(uploaded.read(), uploaded.name)

        if df is not None:
            st.session_state.df = df
            st.success(f"✅ Loaded **{uploaded.name}** — "
                       f"{df.shape[0]:,} rows × {df.shape[1]} columns")

            section("Preview (first 8 rows)")
            st.dataframe(df.head(8), use_container_width=True)

            section("Column Info")
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Dtype": df.dtypes.astype(str).values,
                "Non-Null": df.notnull().sum().values,
                "Missing %": (df.isnull().mean() * 100).round(1).values,
                "Unique": df.nunique().values,
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)

            st.info("👈 Proceed to **Exploratory Analysis** in the sidebar.")
    else:
        st.markdown(
            """<div style="background:#1A1A35;border:1px solid #2E2E55;
                           border-radius:12px;padding:32px;text-align:center;
                           color:#9090B0;">
               <div style="font-size:3rem;">📂</div>
               <div style="margin-top:8px;font-size:1.1rem;">No file uploaded yet.</div>
               <div style="font-size:.8rem;margin-top:4px;">
                 Supported formats: CSV, XLSX, XLS
               </div>
             </div>""",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Page: EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == pages[1]:
    df = st.session_state.df
    if df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    st.markdown('<div class="hero-title" style="font-size:2rem;">🔍 Exploratory Analysis</div>',
                unsafe_allow_html=True)

    # Overview metrics
    overview = eda.dataset_overview(df)
    st.session_state.eda_overview = overview

    section("Dataset Overview")
    cols = st.columns(5)
    with cols[0]: metric_card(f"{overview['rows']:,}", "Rows")
    with cols[1]: metric_card(f"{overview['columns']}", "Columns")
    with cols[2]: metric_card(f"{overview['duplicates']}", "Duplicates")
    with cols[3]: metric_card(f"{overview['total_missing']}", "Missing Cells")
    with cols[4]: metric_card(f"{overview['memory_mb']} MB", "Memory")

    # Column summary
    section("Column Summary")
    col_sum = eda.column_summary(df)
    st.dataframe(col_sum, use_container_width=True, hide_index=True)

    # Descriptive stats
    with st.expander("📊 Descriptive Statistics", expanded=False):
        st.dataframe(eda.descriptive_stats(df), use_container_width=True)

    # Visualisations
    section("Visualisations")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🌡 Correlation", "📈 Distributions", "📋 Categories", "❓ Missing"]
    )

    with tab1:
        fig = eda.plot_correlation_heatmap(df)
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for a correlation heatmap.")

    with tab2:
        fig = eda.plot_distributions(df)
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No numeric columns found.")

    with tab3:
        fig = eda.plot_value_counts(df)
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No categorical columns found.")

    with tab4:
        fig = eda.plot_missing_values(df)
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.success("🎉 No missing values in this dataset!")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
elif page == pages[2]:
    df = st.session_state.df
    if df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    st.markdown('<div class="hero-title" style="font-size:2rem;">⚙️ Preprocessing</div>',
                unsafe_allow_html=True)

    section("Target Column Selection")
    target = st.selectbox(
        "Select the **target** (label) column you want to predict:",
        df.columns.tolist(),
        index=len(df.columns) - 1,
    )
    st.session_state.target = target

    # Auto-detect problem type
    detected = prep.detect_problem_type(df, target)
    st.session_state.problem_type = detected

    col_l, col_r = st.columns([1, 2])
    with col_l:
        badge(f"Detected: {detected.upper()}", "green" if detected == "classification" else "orange")

    with col_r:
        # Allow manual override
        override = st.radio(
            "Override problem type:",
            ["Auto-detect", "Classification", "Regression"],
            horizontal=True,
        )
        if override != "Auto-detect":
            st.session_state.problem_type = override.lower()

    # Show target distribution
    section("Target Distribution")
    fig = eda.plot_target_distribution(df, target)
    st.pyplot(fig, use_container_width=True)

    section("Preprocessing Settings")
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Missing: median (numeric) / mode (categorical)  |  OHE ≤ 20 unique  |  StandardScaler")

    if st.button("🚀 Run Preprocessing Pipeline"):
        with st.spinner("Preprocessing…"):
            try:
                preprocessor = prep.Preprocessor(
                    problem_type=st.session_state.problem_type,
                    test_size=test_size,
                )
                X_train, X_test, y_train, y_test = preprocessor.fit_transform(
                    df, target
                )
                st.session_state.preprocessor = preprocessor
                st.session_state.X_train = X_train
                st.session_state.X_test  = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test  = y_test

                st.success("✅ Preprocessing complete!")

                rep = preprocessor.report
                c1, c2, c3, c4 = st.columns(4)
                with c1: metric_card(rep["train_samples"], "Train Samples")
                with c2: metric_card(rep["test_samples"],  "Test Samples")
                with c3: metric_card(rep["final_feature_count"], "Features")
                with c4: metric_card(rep["duplicates_removed"], "Dupes Removed")

                with st.expander("📋 Preprocessing Details"):
                    st.json(rep)

            except Exception as e:
                st.error(f"❌ Preprocessing failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Train & Evaluate
# ─────────────────────────────────────────────────────────────────────────────
elif page == pages[3]:
    for k in ("X_train", "X_test", "y_train", "y_test", "preprocessor"):
        if st.session_state[k] is None:
            st.warning("⚠️ Please complete Preprocessing first.")
            st.stop()

    st.markdown('<div class="hero-title" style="font-size:2rem;">🤖 Train & Evaluate</div>',
                unsafe_allow_html=True)

    problem_type = st.session_state.problem_type
    badge(problem_type.upper(), "green" if problem_type == "classification" else "orange")

    models_info = {
        "classification": ["Logistic Regression", "Random Forest"],
        "regression":     ["Linear Regression",   "Random Forest"],
    }
    st.markdown(
        f"**Models to train:** {' · '.join(models_info[problem_type])}",
        unsafe_allow_html=True,
    )

    if st.button("▶ Train All Models"):
        with st.spinner("Training…"):
            try:
                trained = mdl.train_models(
                    st.session_state.X_train,
                    st.session_state.y_train,
                    problem_type,
                )
                results, best_name = mdl.evaluate_models(
                    trained,
                    st.session_state.X_test,
                    st.session_state.y_test,
                    problem_type,
                )
                st.session_state.trained_models = trained
                st.session_state.results        = results
                st.session_state.best_name      = best_name
                st.success("✅ Training complete!")

            except Exception as e:
                st.error(f"❌ Training failed: {e}")

    if st.session_state.results is not None:
        results     = st.session_state.results
        best_name   = st.session_state.best_name
        trained     = st.session_state.trained_models
        preprocessor = st.session_state.preprocessor

        # ── Best model banner ────────────────────────────────────────────────
        st.markdown(
            f"""<div class="best-model-banner">
                  <div class="best-model-title">🏆 Best Model: {best_name}</div>
                  <div style="color:#9090B0;font-size:.85rem;margin-top:4px;">
                    Selected by highest {"F1 Score" if problem_type=="classification" else "R² Score"}
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )

        # ── Metrics table ────────────────────────────────────────────────────
        section("Performance Metrics")
        display = clean_results_for_display(results)
        metrics_df = pd.DataFrame(display).T.reset_index().rename(
            columns={"index": "Model"}
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # ── Comparison chart ─────────────────────────────────────────────────
        fig = mdl.plot_metrics_comparison(results, problem_type)
        st.pyplot(fig, use_container_width=True)

        # ── Classification extras ────────────────────────────────────────────
        if problem_type == "classification":
            section("Confusion Matrices")
            cm_cols = st.columns(len(results))
            for i, (name, m) in enumerate(results.items()):
                with cm_cols[i]:
                    st.markdown(f"**{name}**")
                    labels = None
                    if preprocessor.target_encoder is not None:
                        labels = preprocessor.target_encoder.classes_.tolist()
                    fig_cm = mdl.plot_confusion_matrix(m["_cm"], labels)
                    st.pyplot(fig_cm, use_container_width=True)

        # ── Regression extras ────────────────────────────────────────────────
        else:
            section("Actual vs Predicted")
            sc_cols = st.columns(len(results))
            for i, (name, m) in enumerate(results.items()):
                with sc_cols[i]:
                    st.markdown(f"**{name}**")
                    fig_sc = mdl.plot_regression_scatter(
                        st.session_state.y_test, m["_y_pred"], name
                    )
                    st.pyplot(fig_sc, use_container_width=True)

        # ── Feature importance ───────────────────────────────────────────────
        section("Feature Importance — Best Model")
        fig_fi = mdl.plot_feature_importance(
            trained[best_name], preprocessor.feature_names
        )
        if fig_fi:
            st.pyplot(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Dashboard (summary)
# ─────────────────────────────────────────────────────────────────────────────
elif page == pages[4]:
    st.markdown('<div class="hero-title" style="font-size:2rem;">📊 Summary Dashboard</div>',
                unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("No data loaded yet. Start with **Upload Dataset**.")
        st.stop()

    df           = st.session_state.df
    results      = st.session_state.results
    best_name    = st.session_state.best_name
    problem_type = st.session_state.problem_type
    preprocessor = st.session_state.preprocessor
    overview     = st.session_state.eda_overview or eda.dataset_overview(df)

    # ── Dataset card ─────────────────────────────────────────────────────────
    section("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card(f"{overview['rows']:,}", "Rows")
    with c2: metric_card(f"{overview['columns']}", "Columns")
    with c3: metric_card(f"{overview['total_missing']}", "Missing Cells")
    with c4: metric_card(f"{overview['duplicates']}", "Duplicates")

    if results is None:
        st.info("Run the pipeline (Preprocessing → Train & Evaluate) to see model results here.")
        st.stop()

    # ── Best model ────────────────────────────────────────────────────────────
    display = clean_results_for_display(results)
    section("Model Leaderboard")
    badge(problem_type.upper(),
          "green" if problem_type == "classification" else "orange")

    metrics_df = pd.DataFrame(display).T.reset_index().rename(
        columns={"index": "Model"}
    )
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown(
        f"""<div class="best-model-banner">
              <div class="best-model-title">🏆 Best Model: {best_name}</div>
            </div>""",
        unsafe_allow_html=True,
    )

    # ── Key metrics ───────────────────────────────────────────────────────────
    best_metrics = {k: v for k, v in display[best_name].items()
                    if not k.startswith("_")}
    cols = st.columns(len(best_metrics))
    for i, (k, v) in enumerate(best_metrics.items()):
        with cols[i]:
            metric_card(v, k)

    # ── Quick charts ─────────────────────────────────────────────────────────
    section("Quick Visualisations")
    left, right = st.columns(2)

    with left:
        st.markdown("**Correlation Heatmap**")
        fig = eda.plot_correlation_heatmap(df)
        if fig:
            st.pyplot(fig, use_container_width=True)

    with right:
        st.markdown("**Model Comparison**")
        fig = mdl.plot_metrics_comparison(results, problem_type)
        st.pyplot(fig, use_container_width=True)

    if preprocessor is not None:
        section("Feature Importance")
        fig_fi = mdl.plot_feature_importance(
            st.session_state.trained_models[best_name],
            preprocessor.feature_names,
        )
        if fig_fi:
            st.pyplot(fig_fi, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Export
# ─────────────────────────────────────────────────────────────────────────────
elif page == pages[5]:
    st.markdown('<div class="hero-title" style="font-size:2rem;">💾 Export</div>',
                unsafe_allow_html=True)

    if st.session_state.trained_models is None:
        st.warning("⚠️ Train models first before exporting.")
        st.stop()

    trained      = st.session_state.trained_models
    best_name    = st.session_state.best_name
    results      = st.session_state.results
    problem_type = st.session_state.problem_type
    overview     = st.session_state.eda_overview or {}
    preprocessor = st.session_state.preprocessor

    section("Download Trained Model")
    model_bytes = mdl.save_model_bytes(trained[best_name])
    st.download_button(
        label=f"⬇ Download Best Model ({best_name}).pkl",
        data=model_bytes,
        file_name=f"best_model_{best_name.replace(' ','_').lower()}.pkl",
        mime="application/octet-stream",
    )

    st.markdown(
        "<div style='color:#9090B0;font-size:.8rem;'>Load with: "
        "<code>import joblib; model = joblib.load('model.pkl')</code></div>",
        unsafe_allow_html=True,
    )

    section("Download Analysis Report (JSON)")
    report_str = mdl.build_report(
        overview,
        preprocessor.report if preprocessor else {},
        results,
        best_name,
        problem_type,
    )
    st.download_button(
        label="⬇ Download Report (.json)",
        data=report_str,
        file_name="automl_report.json",
        mime="application/json",
    )

    section("Download Results Table (CSV)")
    display = clean_results_for_display(results)
    csv_str = pd.DataFrame(display).T.to_csv()
    st.download_button(
        label="⬇ Download Metrics (.csv)",
        data=csv_str,
        file_name="model_metrics.csv",
        mime="text/csv",
    )

    with st.expander("📋 Report Preview"):
        st.code(report_str[:3000] + ("\n…" if len(report_str) > 3000 else ""),
                language="json")
