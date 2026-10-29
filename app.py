
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------
# 1. Page setup
# -------------------------------
st.set_page_config(
    page_title="Air Quality Monitor",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Air Quality Anomaly Monitor")
st.markdown(
    "### Explainable Air-Quality Anomaly Detection Using MADE"
)

st.info(
    "This application identifies unusual patterns in air-quality "
    "readings. A flagged reading is a candidate for investigation, "
    "not proof of unsafe air or a confirmed pollution event."
)

# -------------------------------
# 2. Load the result files
# -------------------------------
st.sidebar.header("Project Data")

folder = Path(__file__).parent

def load_csv(filename, label):
    default_path = folder / filename

    uploaded = st.sidebar.file_uploader(
        f"Upload {label}",
        type=["csv"],
        key=filename
    )

    if uploaded is not None:
        return pd.read_csv(uploaded)

    if default_path.exists():
        return pd.read_csv(default_path)

    return None


scores_df = load_csv(
    "test_anomaly_scores.csv",
    "anomaly scores CSV"
)

features_df = load_csv(
    "feature_nll_explanations.csv",
    "feature explanations CSV"
)

experiments_df = load_csv(
    "hyperparameter_results.csv",
    "hyperparameter results CSV"
)

# -------------------------------
# 3. Check the required data
# -------------------------------
if scores_df is None:
    st.warning(
        "Please put test_anomaly_scores.csv in the same folder "
        "as app.py, or upload it using the sidebar."
    )
    st.stop()

required_columns = {"anomaly_score", "is_anomaly"}

if not required_columns.issubset(scores_df.columns):
    st.error(
        "The anomaly scores CSV must contain "
        "'anomaly_score' and 'is_anomaly' columns."
    )
    st.stop()

scores_df = scores_df.copy()

scores_df["anomaly_score"] = pd.to_numeric(
    scores_df["anomaly_score"], errors="coerce"
)

scores_df = scores_df.dropna(subset=["anomaly_score"]).reset_index(
    drop=True
)

scores_df["is_anomaly"] = (
    scores_df["is_anomaly"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["1", "true", "yes"])
)

if scores_df.empty:
    st.error("The anomaly scores file has no usable observations.")
    st.stop()

# Load the exact threshold calculated by the notebook
threshold_path = folder / "threshold.txt"

if threshold_path.exists():
    threshold = float(
        threshold_path.read_text().strip()
    )
else:
    st.error(
        "threshold.txt is missing. "
        "Save the threshold from your notebook first."
    )
    st.stop()

# -------------------------------
# 4. Sidebar filters
# -------------------------------
st.sidebar.header("Filters")

show_flagged_only = st.sidebar.checkbox(
    "Show only flagged observations",
    value=False
)

filtered_df = scores_df.copy()

if show_flagged_only:
    filtered_df = filtered_df[
        filtered_df["is_anomaly"]
    ]

# -------------------------------
# 5. Summary metrics
# -------------------------------
total = len(scores_df)
flagged = int(scores_df["is_anomaly"].sum())
percentage = 100 * flagged / total if total else 0

col1, col2, col3 = st.columns(3)

col1.metric("Test observations", f"{total:,}")
col2.metric("Flagged observations", f"{flagged:,}")
col3.metric("Flagged percentage", f"{percentage:.2f}%")

# -------------------------------
# 6. Dashboard tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔎 Investigate",
    "🧩 Feature Explanations",
    "⚙️ Model Experiments"
])

# -------------------------------
# TAB 1: Overview
# -------------------------------
with tab1:
    st.subheader("Anomaly score across test observations")

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(
        scores_df["anomaly_score"].to_numpy(),
        label="Anomaly score",
        linewidth=0.8
    )

    ax.axhline(
        threshold,
        linestyle="--",
        label="Approximate displayed threshold"
    )

    ax.set_xlabel("Test observation index")
    ax.set_ylabel("Average negative log-likelihood")
    ax.legend()
    ax.grid(alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Distribution of anomaly scores")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.hist(scores_df["anomaly_score"], bins=35)
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Number of observations")
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "The displayed threshold is estimated from the exported "
        "flags. For the exact threshold, use the value printed "
        "by the notebook."
    )

# -------------------------------
# TAB 2: Investigate
# -------------------------------
with tab2:
    st.subheader("Inspect observations")

    display_df = filtered_df.copy()
    display_df.insert(
        0, "test_index", display_df.index
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    if not filtered_df.empty:
        selected_index = st.selectbox(
            "Select a test observation",
            filtered_df.index.tolist(),
            format_func=lambda i: (
                f"Observation {i} | "
                f"Score: "
                f"{filtered_df.loc[i, 'anomaly_score']:.3f} | "
                f"{'FLAGGED' if filtered_df.loc[i, 'is_anomaly'] else 'Not flagged'}"
            )
        )

        selected_row = filtered_df.loc[selected_index]

        st.metric(
            "Selected observation score",
            f"{selected_row['anomaly_score']:.4f}"
        )

        if selected_row["is_anomaly"]:
            st.warning(
                "This observation was flagged by the model "
                "and should be investigated."
            )
        else:
            st.success(
                "This observation was not flagged by the model."
            )

# -------------------------------
# TAB 3: Feature explanations
# -------------------------------
with tab3:
    st.subheader("Which features contributed most?")

    if features_df is None:
        st.warning(
            "Upload feature_nll_explanations.csv or place it "
            "beside app.py."
        )

    elif len(features_df) != len(scores_df):
        st.error(
            "The feature explanation file and anomaly score file "
            "must contain the same number of test observations "
            "in the same order."
        )

    else:
        feature_columns = list(features_df.columns)

        selected_anomaly = st.selectbox(
            "Choose an observation to explain",
            range(len(scores_df)),
            format_func=lambda i: (
                f"Observation {i} | "
                f"Score: {scores_df.loc[i, 'anomaly_score']:.3f} | "
                f"{'FLAGGED' if scores_df.loc[i, 'is_anomaly'] else 'Not flagged'}"
            )
        )

        contributions = pd.to_numeric(
            features_df.iloc[selected_anomaly],
            errors="coerce"
        ).dropna().sort_values(ascending=False)

        explanation_df = contributions.rename(
            "Feature NLL contribution"
        ).reset_index()

        explanation_df.columns = [
            "Feature",
            "Feature NLL contribution"
        ]

        st.dataframe(
            explanation_df,
            use_container_width=True,
            hide_index=True
        )

        fig, ax = plt.subplots(figsize=(10, 4))

        plot_df = explanation_df.sort_values(
            "Feature NLL contribution"
        )

        ax.barh(
            plot_df["Feature"],
            plot_df["Feature NLL contribution"]
        )

        ax.set_xlabel("Feature NLL contribution")
        ax.set_title("Contribution to the selected observation's score")
        ax.grid(axis="x", alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Higher contributions indicate features that contributed "
            "more to the model's score. They do not prove the cause "
            "of an unusual event."
        )

# -------------------------------
# TAB 4: Model experiments
# -------------------------------
with tab4:
    st.subheader("Hyperparameter comparison")

    if experiments_df is None:
        st.warning(
            "Upload hyperparameter_results.csv or place it "
            "beside app.py."
        )

    else:
        st.dataframe(
            experiments_df,
            use_container_width=True,
            hide_index=True
        )

        if "validation_nll" in experiments_df.columns:
            chart_df = experiments_df.sort_values(
                "validation_nll"
            ).reset_index(drop=True)

            st.markdown(
                "Lower validation NLL is generally better "
                "for comparing these configurations."
            )

            fig, ax = plt.subplots(figsize=(10, 4))

            labels = [
                f"Experiment {i + 1}"
                for i in range(len(chart_df))
            ]

            ax.bar(
                labels,
                chart_df["validation_nll"]
            )

            ax.set_ylabel("Validation NLL")
            ax.set_xlabel("Experiment")
            ax.grid(axis="y", alpha=0.2)

            st.pyplot(fig)
            plt.close(fig)

        else:
            st.info(
                "The uploaded experiment file has no "
                "'validation_nll' column."
            )

# -------------------------------
# Footer
# -------------------------------
st.divider()

st.caption(
    "MADE = Masked Autoencoder for Distribution Estimation | "
    "Air-quality anomaly screening project"
)
