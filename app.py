import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import base64
import os

st.set_page_config(page_title="Future Stars", layout="wide")


# ---------------- Background image (robust) ----------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

# put your stadium image here:  images/bg.jpg   (falls back to a gradient if missing)
BG_CANDIDATES = ["images/bg.jpg", "images/pitch_pic.JPG", "images/pitch_pic.jpg"]
base64_image = None
for _p in BG_CANDIDATES:
    if os.path.exists(_p):
        base64_image = get_base64_of_bin_file(_p)
        break

# Custom CSS
st.markdown("""
<style>
/* ---------------- Tabs ---------------- */
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    border-bottom: 3px solid #84A98C !important;
    color: #84A98C !important;
    font-weight: bold;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="false"] {
    color: #CAD2C5 !important;
}

/* ---------------- Sidebar ---------------- */
[data-testid="stSidebar"] {
    min-width: 280px;
    max-width: 320px;
    background:#23343a;
}
[data-testid="stSidebar"] * {
    color: #CAD2C5 !important;
}

/* ---------------- Sliders ---------------- */
.stSlider [role="slider"] {
    background: #84A98C !important;
    border: 2px solid #84A98C !important;
}

/* ---------------- KPI Grid + Cards ---------------- */
.kpi-grid{ display: grid; grid-template-columns: 1fr 1fr; grid-gap: 20px; height: 100%; }
.kpi-card{
  background: linear-gradient(145deg, #2F3E46, #354F52);
  border-radius: 16px; padding: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.35);
  display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;
}
.kpi-title{ font-size:14px; color:#CAD2C5; opacity:.9; margin-bottom:6px; }
.kpi-value{ font-size:26px; font-weight:800; color:#84A98C; }
.kpi-sub{ font-size:12px; color:#CAD2C5; opacity:.7; margin-top:4px; }

/* ---------------- Container Cards ---------------- */
section[data-testid="stContainer"] {
    border-radius: 16px; padding: 20px;
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(6px);
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
    margin-bottom: 1rem;
}

/* ---------------- Headings ---------------- */
h3, h4 { font-weight: 600; color: #CAD2C5 !important; }

/* ---------------- Title Styling ---------------- */
h1 {
    position: relative; font-size: 3.5em; font-weight: 900; text-align: center;
    background: linear-gradient(90deg, #84A98C, #1DB954);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-transform: uppercase; letter-spacing: 2px;
    text-shadow: 0px 0px 10px rgba(0,0,0,0.6); margin-bottom: 0.5em;
}
h1::before {
    content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.6), transparent);
    animation: shine 3s infinite;
}
@keyframes shine { 0% { left: -100%; } 50% { left: 100%; } 100% { left: 100%; } }

h2, h3, p {
    text-align: center; color: #CAD2C5 !important; font-size: 1.1em;
    font-style: italic; opacity: 0.9; margin-top: -0.5em;
}
hr { border: 0; height: 1px; background: rgba(255,255,255,0.25); margin: 1em auto; width: 50%; border-radius: 2px; }

/* ---------------- Predict button ---------------- */
div.stButton > button {
    background: linear-gradient(90deg, #1DB954, #1ED760); color: white;
    font-size: 18px; font-weight: bold; border-radius: 30px; padding: 0.6em 2em;
}
div.stButton > button:hover { background: linear-gradient(90deg, #17a74a, #1db954); transform: scale(1.05); }
</style>
""", unsafe_allow_html=True)

# ---- Self-contained: load the model and predict directly (no separate API needed) ----
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from future_stars.predict import predict

MODEL_PATH = str(ROOT / "model" / "model.pkl")


def score_players(df: pd.DataFrame) -> pd.DataFrame:
    res = predict(data=df, model_path=MODEL_PATH)
    if "Probability" in res.columns:
        res["Probability"] = pd.to_numeric(
            res["Probability"].astype(str).str.replace("%", "", regex=False), errors="coerce")
    return res


def player_row(name, nat, age, pos, minutes, ga, xg, xag, tkl, blocks, clr,
               prgp, kp, prgc, prgr, shots, sot, savep) -> pd.DataFrame:
    minutes = max(float(minutes), 1.0)
    n90 = minutes / 90.0
    return pd.DataFrame([{
        "Player": name, "Nation": nat or "NA", "Age": age, "Pos": pos,
        "Min": minutes, "90s": n90, "G+A": ga, "xG": xg, "xAG": xag,
        "Tkl": tkl, "Blocks_stats_defense": blocks, "Clr": clr,
        "PrgP": prgp, "KP": kp, "PrgC": prgc, "PrgR": prgr,
        "Sh/90": shots / n90 if n90 else 0.0, "SoT/90": sot / n90 if n90 else 0.0,
        "Save%": savep,
    }])


earthy_colors = {"light": "#CAD2C5", "green": "#84A98C", "teal": "#52796F",
                 "deep": "#354F52", "dark": "#2F3E46"}

if not (ROOT / "model" / "model.pkl").exists():
    st.error("Model not found at model/model.pkl. Train it first (python main.py) and include it in the repo.")
    st.stop()

# Navigation
page = st.sidebar.radio("Navigation", ["Predict Player", "Analysis Dashboard"])

# ============================ Page 1: Predict Player ============================
if page == "Predict Player":
    # ---------------- Background ----------------
    if base64_image:
        bg_layer = (f'linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.2)), '
                    f'url("data:image/jpg;base64,{base64_image}")')
    else:
        bg_layer = ("radial-gradient(1200px 600px at 50% -10%, #14502f 0%, "
                    "#0b2a1a 45%, #07120c 100%)")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {bg_layer};
            background-size: cover; background-position: center;
            background-repeat: no-repeat; background-attachment: fixed;
        }}
        h1, h2, h3 {{ text-shadow: 2px 2px 8px rgba(0,0,0,0.8); }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("# Are You the Next Future Star")
    st.markdown("Welcome to the **Future Stars Talent Predictor**! Enter your details below to check your potential.")

    st.subheader("Player Input")
    with st.form("player_form"):
        col1, col2, col3 = st.columns(3)
        player_name = col1.text_input("Player Name", placeholder="Enter your name")
        age = col2.number_input("Age", max_value=45)
        positions = ["GK", "DF", "MF", "FW"]
        pos = col3.selectbox("Position", ["Select a Position"] + positions, index=0)

        col4, col5, col6 = st.columns(3)
        minutes = col4.number_input("Minutes Played", min_value=0)
        ga = col5.number_input("Goals + Assists", min_value=0)
        xg = col6.number_input("Expected Goals (xG)", min_value=0.0, step=0.1)

        col7, col8, col9 = st.columns(3)
        xag = col7.number_input("Expected Assists (xAG)", min_value=0.0, step=0.1)
        prog_pass = col8.number_input("Progressive Passes", min_value=0)
        save_pct = col9.number_input("Save % (GK only)", min_value=0.0, max_value=1.0, step=0.01)

        col10, col11, col12 = st.columns(3)
        tackles = col10.number_input("Tackles", min_value=0)
        blocks = col11.number_input("Blocks", min_value=0)
        clearances = col12.number_input("Clearances", min_value=0)

        # extra stats the temporal model uses (optional -> better accuracy)
        with st.expander("More stats (optional — improves accuracy)"):
            e1, e2, e3 = st.columns(3)
            key_passes = e1.number_input("Key Passes", min_value=0)
            prog_carries = e2.number_input("Progressive Carries", min_value=0)
            prog_runs = e3.number_input("Progressive Runs received", min_value=0)
            e4, e5 = st.columns(2)
            shots = e4.number_input("Shots", min_value=0)
            shots_on_target = e5.number_input("Shots on Target", min_value=0)

        nationality = st.text_input("Nationality", placeholder="Enter your nationality")

        b1, b2, b3 = st.columns([2, 1, 2])
        with b2:
            submitted = st.form_submit_button("Predict")

    if submitted:
        with st.spinner("Scoring..."):
            try:
                df_result = score_players(player_row(
                    player_name, nationality, age, pos, minutes, ga, xg, xag,
                    tackles, blocks, clearances, prog_pass, key_passes,
                    prog_carries, prog_runs, shots, shots_on_target, save_pct))
                st.success("Prediction Complete")
                st.write(df_result)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # ---------------- CSV Upload ----------------
    st.subheader("Or upload a CSV file for multiple players")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        if st.button("Show Analysis"):
            with st.spinner("Processing..."):
                try:
                    raw = pd.read_csv(uploaded_file)
                    df_pred = score_players(raw)
                except Exception as e:
                    st.error(f"Could not analyze this file: {e}")
                    df_pred = None
            if df_pred is not None:
                st.session_state["analysis_df"] = df_pred
                st.success(f"Predictions ready for {len(df_pred)} players! "
                           f"Full table below — or open the Analysis Dashboard for charts.")
                show = df_pred.copy()
                if "Probability" in show.columns:
                    show = show.sort_values("Probability", ascending=False)
                st.dataframe(show, use_container_width=True, hide_index=True)
                st.download_button("Download results (CSV)",
                                   show.to_csv(index=False), "future_star_scores.csv", "text/csv")


# ============================ Page 2: Analysis Dashboard ============================
if page == "Analysis Dashboard":
    # ---- Same stadium background, stronger dark overlay so charts/tables stay readable ----
    if base64_image:
        dash_bg = (f'linear-gradient(rgba(7,18,12,0.90), rgba(7,18,12,0.94)), '
                   f'url("data:image/jpg;base64,{base64_image}")')
    else:
        dash_bg = ("radial-gradient(1200px 600px at 50% -10%, #14502f 0%, "
                   "#0b2a1a 45%, #07120c 100%)")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {dash_bg};
            background-size: cover; background-position: center;
            background-repeat: no-repeat; background-attachment: fixed;
        }}
        h1, h2, h3 {{ text-shadow: 2px 2px 8px rgba(0,0,0,0.85); }}
        /* make KPI metrics readable over the image */
        [data-testid="stMetric"] {{
            background: rgba(33, 51, 46, 0.55);
            border: 1px solid rgba(132,169,140,0.25);
            border-radius: 14px; padding: 12px 16px;
            backdrop-filter: blur(4px);
        }}
        [data-testid="stMetricValue"] {{ color:#84A98C !important; }}
        [data-testid="stMetricLabel"] {{ color:#CAD2C5 !important; }}
        /* tabs + dataframe readable over the image */
        [data-testid="stDataFrame"] {{
            background: rgba(20,30,26,0.65); border-radius: 12px; padding: 4px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Analysis Dashboard")

    if "analysis_df" in st.session_state:
        df = st.session_state["analysis_df"]

        if "Probability" in df.columns:
            df["Probability"] = pd.to_numeric(
                df["Probability"].astype(str).str.replace("%", "", regex=False), errors="coerce")
        elif "probability" in df.columns:
            df["Probability"] = (df["probability"].astype(float) * 100).round(2)

        # Sidebar Filters
        with st.sidebar:
            st.markdown("### Filters")
            pos_options = sorted(df["Pos"].dropna().unique().tolist())
            pred_options = sorted(df["Prediction"].dropna().unique().tolist())
            pos_filter = st.multiselect("Position", pos_options, default=pos_options)
            pred_filter = st.multiselect("Prediction", pred_options, default=pred_options)
            age_range = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (18, 30))
            prob_range = st.slider("Probability (%)", 0.0, 100.0, (0.0, 100.0))

        # Apply filters
        fdf = df.copy()
        fdf = fdf[fdf["Pos"].isin(pos_filter)]
        fdf = fdf[fdf["Prediction"].isin(pred_filter)]
        fdf = fdf[fdf["Age"].between(age_range[0], age_range[1])]
        fdf = fdf[fdf["Probability"].between(prob_range[0], prob_range[1])]

        tab1, tab2 = st.tabs(["Visual Analysis", "All Players Stats"])

        with tab1:
            # Row 1: KPI Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Age AVG", round(df["Age"].mean(), 1))
            c2.metric("Total Players", len(df))
            c3.metric("Avg Probability", f"{df['Probability'].mean():.2f}%")
            c4.metric("Future Stars %", f"{(df['Prediction'].eq('Future Star').mean()*100):.1f}%")
            st.markdown("---")

            # Row 2: Top Future Star + Bar Chart
            left_col, right_col = st.columns([1, 2])
            if not fdf.empty:
                top_player = fdf.sort_values("Probability", ascending=False).iloc[0]
                with left_col.container(border=True):
                    st.markdown("### Top Future Star")
                    st.markdown(f"**{top_player['Player']}**")
                    st.markdown(f"{top_player['Pos']} | {int(top_player['Age'])} yrs")
                    st.markdown(f"**{top_player['Probability']:.2f}%**")
                    st.markdown(f"Key Metric: {top_player.get('Key Metric', '')}")

            by_pos_pred = fdf.groupby(["Pos", "Prediction"]).size().reset_index(name="Count")
            fig1 = px.bar(by_pos_pred, x="Pos", y="Count", color="Prediction", barmode="group",
                          title="Future Stars vs Not Future Stars by Position",
                          color_discrete_sequence=[earthy_colors["green"], earthy_colors["teal"]])
            fig1.update_layout(plot_bgcolor=earthy_colors["dark"], paper_bgcolor=earthy_colors["dark"],
                               font=dict(color=earthy_colors["light"]))
            with right_col.container(border=True):
                st.plotly_chart(fig1, use_container_width=True)

            # Row 3: Top 5 Bar Chart + Scatter
            row3_col1, row3_col2 = st.columns([1.2, 1])
            if not fdf.empty:
                top5 = fdf.sort_values("Probability", ascending=False).head(5)
                fig_bar = px.bar(top5, x="Probability", y="Player", orientation="h", text="Probability",
                                 title="Top 5 Future Stars by Probability",
                                 color_discrete_sequence=[earthy_colors["green"]])
                fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig_bar.update_layout(xaxis_title="Probability (%)", yaxis_title="",
                                      yaxis=dict(autorange="reversed"), height=400)
                with row3_col1.container(border=True):
                    st.plotly_chart(fig_bar, use_container_width=True)

                fig_scatter = px.scatter(fdf, x="Age", y="Probability", color="Prediction",
                                         hover_name="Player", title="Age vs Probability", size="Probability",
                                         color_discrete_sequence=[earthy_colors["green"], earthy_colors["teal"]])
                with row3_col2.container(border=True):
                    st.plotly_chart(fig_scatter, use_container_width=True)

        # ---- tab2 now at the correct level (was wrongly nested before) ----
        with tab2:
            st.markdown("### All Players Stats")
            st.dataframe(fdf, use_container_width=True)

    else:
        st.warning("No predictions loaded yet. Please upload CSV on 'Predict Player' page.")
