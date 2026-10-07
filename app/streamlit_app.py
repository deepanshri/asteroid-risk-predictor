"""Streamlit app: predict whether an asteroid close approach is a PHA, with SHAP explanation.

Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `src` is importable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from src.config import CLEAN_CSV, MODEL_PATH, RAW_FEATURES
from src.features import LUNAR_DISTANCE_KM, add_features, diameter_km_from_h

st.set_page_config(page_title="Asteroid Hazard Predictor", page_icon="☄️", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV)


if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

bundle = load_model()
pipe, threshold, names = bundle["pipeline"], bundle["threshold"], bundle["features"]
data = load_dataset()

st.title("☄️ Asteroid Hazard Predictor")
tab_predict, tab_data, tab_about = st.tabs(["Predict", "Dataset explorer", "About"])

# ---------------------------------------------------------------- inputs ----
with st.sidebar:
    st.header("Close-approach parameters")
    h = st.slider("Absolute magnitude H (lower = bigger)", 9.0, 33.0, 21.0, 0.1)
    speed = st.slider("Relative velocity (km/s)", 0.1, 60.0, 15.0, 0.1)
    miss_ld = st.slider("Miss distance (lunar distances)", 0.01, 195.0, 20.0, 0.01)
    st.caption(f"≈ {diameter_km_from_h(h) * 1000:,.0f} m wide (albedo 0.15 assumed) · "
               f"{miss_ld * LUNAR_DISTANCE_KM:,.0f} km miss distance")

row = pd.DataFrame([{"absolute_magnitude": h, "relative_velocity": speed * 3600,
                     "miss_distance": miss_ld * LUNAR_DISTANCE_KM}])[RAW_FEATURES]
proba = float(pipe.predict_proba(row)[0, 1])
is_pha = proba >= threshold

# --------------------------------------------------------------- predict ----
with tab_predict:
    c1, c2, c3 = st.columns(3)
    c1.metric("Hazard probability", f"{proba:.1%}")
    c2.metric("Decision threshold", f"{threshold:.2f}")
    c3.metric("Verdict", "⚠️ Potentially hazardous" if is_pha else "✅ Not flagged")
    st.caption("The threshold is deliberately low-ish: it is tuned to catch ~90% of real "
               "PHAs, so expect false alarms (see About).")

    st.subheader("Why did the model say that?")
    Z = pd.DataFrame(pipe[:-1].transform(row), columns=names)
    sv = shap.TreeExplainer(pipe.named_steps["clf"])(Z)
    vals = sv.values[0, :, 1] if sv.values.ndim == 3 else sv.values[0]
    base = float(np.ravel(sv.base_values)[0])
    shown = add_features(row)[names].iloc[0].round(2)
    shap.force_plot(base, vals, shown, matplotlib=True, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Red pushes the prediction toward 'hazardous', blue toward 'not hazardous' "
               "(log-odds scale).")

# ----------------------------------------------------------- dataset tab ----
with tab_data:
    st.write("Sample of dataset close approaches with your input highlighted (absolute "
             "magnitude vs miss distance).")
    sample = data.sample(8000, random_state=0).assign(
        miss_ld=lambda d: d.miss_distance / LUNAR_DISTANCE_KM,
        label=lambda d: d.hazardous.map({0: "Not PHA", 1: "PHA"}))
    fig = px.scatter(sample, x="absolute_magnitude", y="miss_ld", color="label", opacity=0.45,
                     log_y=True, color_discrete_map={"Not PHA": "#9aa5b1", "PHA": "#d64545"},
                     labels={"absolute_magnitude": "Absolute magnitude H",
                             "miss_ld": "Miss distance (lunar distances)"})
    fig.add_scatter(x=[h], y=[miss_ld], mode="markers", name="Your input",
                    marker=dict(size=16, color="gold", line=dict(width=2, color="black"),
                                symbol="star"))
    fig.add_vline(x=22, line_dash="dash", annotation_text="H = 22")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("PHAs only exist at H ≤ 22 (left of the dashed definition boundary), "
               "but many bright rocks there are not hazardous.")

# ----------------------------------------------------------------- about ----
with tab_about:
    st.markdown("""
**What is a PHA?** NASA calls an asteroid *Potentially Hazardous* if its orbit passes within
**0.05 AU (~7.5 million km)** of Earth's orbit (MOID ≤ 0.05 AU) **and** it is bigger than about
140 m (absolute magnitude **H ≤ 22**).

**What this model sees.** Only three numbers from a single close approach: H, relative velocity
and miss distance. The dataset has no orbital elements and no MOID, which is the quantity that
actually defines a PHA.

**Limits**
- H is half of the definition, so most of the signal is "is it big enough?". Flyby speed and
  distance only weakly hint at the orbit.
- Precision is low (~1 in 3 flagged rows is a real PHA) because the threshold favours recall:
  missing a hazardous asteroid is worse than a false alarm.
- Sizes are estimates from H; the true albedo is unknown.
- Educational project, **not** a planetary-defence tool. Official lists: NASA JPL CNEOS.
""")
