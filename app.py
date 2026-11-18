import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.ndimage import gaussian_filter1d

DEFAULT_THRESHOLD = 0.2

st.set_page_config(
    page_title="Exoplanet Detector",
    page_icon="🪐",
    layout="wide",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1200px;}
    .hero {
        background: linear-gradient(120deg, #0b1d3a 0%, #1b3a6b 55%, #3b2f7a 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 18px;
    }
    .hero h1 {margin: 0 0 6px 0; font-size: 2.1rem; color: #ffffff;}
    .hero p {margin: 0; font-size: 1.02rem; color: #d6e2ff;}
    div[data-testid="stMetric"] {
        background: rgba(120, 140, 200, 0.10);
        border: 1px solid rgba(120, 140, 200, 0.25);
        padding: 14px 16px;
        border-radius: 12px;
    }
    .note {
        font-size: 0.9rem;
        opacity: 0.85;
        border-left: 4px solid #6c8cff;
        padding: 6px 12px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Model and features (same as training) ----------
def make_features(X):
    f = pd.DataFrame()
    f["min"] = X.min(axis=1)
    f["max"] = X.max(axis=1)
    f["std"] = X.std(axis=1)
    f["p01"] = np.percentile(X, 1, axis=1)
    f["p05"] = np.percentile(X, 5, axis=1)
    f["p95"] = np.percentile(X, 95, axis=1)
    f["dips_2"] = (X < -2).sum(axis=1)
    f["dips_3"] = (X < -3).sum(axis=1)
    f["dips_4"] = (X < -4).sum(axis=1)
    f["diff_std"] = np.diff(X, axis=1).std(axis=1)
    f["diff_min"] = np.diff(X, axis=1).min(axis=1)
    f["diff_max"] = np.diff(X, axis=1).max(axis=1)
    return f


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


@st.cache_data(show_spinner=False)
def score_stars(file_bytes):
    import io

    data = pd.read_csv(io.BytesIO(file_bytes))
    flux_cols = [c for c in data.columns if c.startswith("FLUX")]
    if not flux_cols:
        return None, None
    flux = data[flux_cols].fillna(0).values.astype(float)
    spread = flux.std(axis=1, keepdims=True)
    spread[spread == 0] = 1.0
    flux = (flux - flux.mean(axis=1, keepdims=True)) / spread
    flux = gaussian_filter1d(flux, sigma=3, axis=1)
    feats = make_features(flux)
    prob = load_model().predict_proba(feats)[:, 1]
    return prob, flux


# ---------- Sidebar ----------
with st.sidebar:
    st.header("🪐 About")
    st.write(
        "This app looks at how a star's brightness changes over time. "
        "When a planet passes in front of its star, the light dips slightly. "
        "A Random Forest model trained on Kaggle exoplanet data flags stars "
        "that may have a planet."
    )
    st.header("How to use")
    st.markdown(
        "1. Upload a CSV with columns `FLUX.1`, `FLUX.2`, ...\n"
        "2. Check the summary numbers\n"
        "3. Explore the table and charts\n"
        "4. Download the results"
    )
    st.header("Sensitivity")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.90,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="A star is flagged as a possible planet when its probability is at "
        "least this value. Lower means more planets found but more false alarms.",
    )
    st.caption("Default 0.2 was chosen to find as many planets as possible.")

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>🪐 Exoplanet Detection</h1>
        <p>Upload star brightness data and let the model flag stars that may host an exoplanet.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

file = st.file_uploader("Upload your CSV file of star flux values", type="csv")

if not file:
    st.info("Upload a CSV file to get started. Use the test file with FLUX columns.")
    st.stop()

with st.spinner("Analysing stars..."):
    prob, flux = score_stars(file.getvalue())

if prob is None:
    st.error("No columns starting with FLUX were found. Please upload the star flux CSV.")
    st.stop()

# ---------- Results ----------
out = pd.DataFrame({"Star": np.arange(1, len(prob) + 1)})
out["Probability"] = prob.round(3)
out["Prediction"] = np.where(prob >= threshold, "Possible exoplanet", "No exoplanet")

total = len(out)
found = int((prob >= threshold).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Stars analysed", f"{total}")
c2.metric("Possible exoplanets", f"{found}")
c3.metric("No exoplanet", f"{total - found}")
c4.metric("Flagged share", f"{found / total:.1%}")

tab1, tab2, tab3 = st.tabs(["📋 Results", "📊 Charts", "🔭 Light curve viewer"])

with tab1:
    show = st.radio(
        "Show", ["All stars", "Possible exoplanets only"], horizontal=True
    )
    view = out if show == "All stars" else out[out["Prediction"] == "Possible exoplanet"]
    view = view.sort_values("Probability", ascending=False)
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Probability": st.column_config.ProgressColumn(
                "Probability of planet", min_value=0.0, max_value=1.0, format="%.3f"
            )
        },
    )
    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Download all results",
        out.to_csv(index=False),
        "results.csv",
        use_container_width=True,
    )
    d2.download_button(
        "⬇️ Download possible exoplanets only",
        out[out["Prediction"] == "Possible exoplanet"].to_csv(index=False),
        "possible_exoplanets.csv",
        use_container_width=True,
    )

with tab2:
    left, right = st.columns(2)
    with left:
        st.subheader("Prediction counts")
        counts = out["Prediction"].value_counts().rename_axis("Result").reset_index(name="Stars")
        st.bar_chart(counts, x="Result", y="Stars")
    with right:
        st.subheader("Probability distribution")
        bins = pd.cut(out["Probability"], bins=np.linspace(0, 1, 11), include_lowest=True)
        hist = bins.value_counts().sort_index()
        hist_df = pd.DataFrame(
            {"Probability range": [f"{i.left:.1f}-{i.right:.1f}" for i in hist.index], "Stars": hist.values}
        )
        st.bar_chart(hist_df, x="Probability range", y="Stars")

with tab3:
    st.write("Pick a star to see its processed brightness curve. Dips can hint at a transit.")
    star_no = st.number_input("Star number", min_value=1, max_value=total, value=int(out.sort_values("Probability", ascending=False).iloc[0]["Star"]), step=1)
    row = out[out["Star"] == star_no].iloc[0]
    st.metric("Probability of planet", f"{row['Probability']:.3f}", row["Prediction"], delta_color="off")
    curve = pd.DataFrame({"Brightness (standardised)": flux[int(star_no) - 1]})
    st.line_chart(curve)

st.markdown(
    '<div class="note">A "Possible exoplanet" is a candidate, not a confirmed planet. '
    "Lower thresholds find more planets but also raise more false alarms.</div>",
    unsafe_allow_html=True,
)
