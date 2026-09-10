import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
from scipy.ndimage import gaussian_filter

st.set_page_config(
    page_title="Opta-Style Tactical Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Sleek Dark Glassmorphism UI
st.markdown(
    """
<style>
    .stApp { background-color: #0b0f19; color: #f1f5f9; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; }
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 14px;
        border-radius: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# 1. Advanced Simulated Event Dataset with Positional Coordinates & Physics
@st.cache_data
def generate_tactical_data():
    np.random.seed(42)
    profiles = [
        {
            "player": "Erling Haaland",
            "team": "Manchester City",
            "pos": "CF",
            "n": 95,
            "loc_x": (88, 102),
            "loc_y": (28, 40),
            "conv": 0.28,
            "foot": "Left",
        },
        {
            "player": "Mohamed Salah",
            "team": "Liverpool",
            "pos": "RW",
            "n": 105,
            "loc_x": (82, 98),
            "loc_y": (38, 56),
            "conv": 0.20,
            "foot": "Left",
        },
        {
            "player": "Bukayo Saka",
            "team": "Arsenal",
            "pos": "RW",
            "n": 88,
            "loc_x": (80, 96),
            "loc_y": (36, 54),
            "conv": 0.18,
            "foot": "Left",
        },
        {
            "player": "Son Heung-min",
            "team": "Tottenham Hotspur",
            "pos": "LW",
            "n": 78,
            "loc_x": (78, 96),
            "loc_y": (16, 32),
            "conv": 0.22,
            "foot": "Right",
        },
        {
            "player": "Cole Palmer",
            "team": "Chelsea",
            "pos": "AM/RW",
            "n": 82,
            "loc_x": (76, 94),
            "loc_y": (22, 46),
            "conv": 0.24,
            "foot": "Left",
        },
        {
            "player": "Phil Foden",
            "team": "Manchester City",
            "pos": "AM/LW",
            "n": 90,
            "loc_x": (78, 95),
            "loc_y": (22, 46),
            "conv": 0.21,
            "foot": "Left",
        },
    ]

    all_records = []
    for p in profiles:
        xs = np.random.uniform(p["loc_x"][0], p["loc_x"][1], p["n"])
        ys = np.random.uniform(p["loc_y"][0], p["loc_y"][1], p["n"])

        # Calculate true geometric distance and visual angle to goal posts (105, 30.66) to (105, 37.34)
        dx = 105.0 - xs
        dy = np.abs(34.0 - ys)
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(7.32 * dx, dx**2 + dy**2 - (7.32 / 2) ** 2)

        # Non-linear logistic xG approximation formula
        logit = -0.15 * dist + 1.2 * angle - 0.8
        xg = 1.0 / (1.0 + np.exp(-logit))
        xg = np.clip(xg * (p["conv"] / 0.16), 0.02, 0.92)

        # Assign outcomes
        sorted_indices = np.argsort(xg)
        goals_count = int(p["n"] * p["conv"])
        goal_indices = set(sorted_indices[-goals_count:])

        for i in range(p["n"]):
            all_records.append(
                {
                    "Player": p["player"],
                    "Team": p["team"],
                    "Position": p["pos"],
                    "Preferred Foot": p["foot"],
                    "X": xs[i],
                    "Y": ys[i],
                    "Distance (m)": np.round(dist[i], 1),
                    "Shot Angle (rad)": np.round(angle[i], 2),
                    "xG": np.round(xg[i], 3),
                    "Outcome": (
                        "Goal"
                        if i in goal_indices
                        else ("Saved" if i % 3 == 0 else "Missed")
                    ),
                }
            )

    return pd.DataFrame(all_records)


df = generate_tactical_data()

# 2. Header & Controls
col_title, col_logo = st.columns([4, 1])
with col_title:
    st.title("⚽ Opta Tactical Analytics Hub")
    st.caption(
        "Spatial Shot Modeling, Continuous Kernel Density Surfaces & Expected Goals (xG) Engine"
    )

st.sidebar.markdown("### Selection & Filter")
selected_team = st.sidebar.selectbox("Club", sorted(df["Team"].unique()))
team_df = df[df["Team"] == selected_team]
selected_player = st.sidebar.selectbox("Player", sorted(team_df["Player"].unique()))

p_df = team_df[team_df["Player"] == selected_player].copy()

# 3. High-Tier KPI Metrics
total_shots = len(p_df)
goals = len(p_df[p_df["Outcome"] == "Goal"])
total_xg = p_df["xG"].sum()
xg_diff = goals - total_xg
xg_per_shot = total_xg / max(total_shots, 1)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Shots", total_shots)
m2.metric("Actual Goals", goals)
m3.metric("Accumulated xG", f"{total_xg:.2f}")
m4.metric("xG Overperformance", f"{xg_diff:+.2f}", delta=f"{xg_diff:+.2f}")
m5.metric("Avg Shot Quality", f"{xg_per_shot:.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# 4. Two Visual Layout: Spatial Scatter + Continuous KDE Heatmap
tab1, tab2 = st.tabs(
    ["🎯 Shot Trajectory & Outcomes", "🔥 Continuous Shot Density (KDE)"]
)

with tab1:
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#0b0f19")
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=105,
        pitch_width=68,
        half=True,
        pitch_color="#0f172a",
        line_color="#334155",
        line_zorder=2,
        goal_type="box",
    )
    pitch.draw(ax=ax)

    # Plot misses and saves
    non_goals = p_df[p_df["Outcome"] != "Goal"]
    pitch.scatter(
        non_goals["X"],
        non_goals["Y"],
        s=non_goals["xG"] * 900 + 40,
        color="#ef4444",
        edgecolors="#ffffff",
        linewidth=0.8,
        alpha=0.6,
        ax=ax,
        label="Missed / Saved",
    )

    # Plot goals with bright gold accent
    goals_df = p_df[p_df["Outcome"] == "Goal"]
    pitch.scatter(
        goals_df["X"],
        goals_df["Y"],
        s=goals_df["xG"] * 1100 + 60,
        color="#38bdf8",
        edgecolors="#ffffff",
        linewidth=1.5,
        alpha=0.95,
        ax=ax,
        label="Goal",
    )

    ax.legend(
        facecolor="#1e293b", edgecolor="none", labelcolor="#f8fafc", loc="upper left"
    )
    st.pyplot(fig, use_container_width=True)

with tab2:
    fig_kde, ax_kde = plt.subplots(figsize=(11, 7), facecolor="#0b0f19")
    pitch_kde = Pitch(
        pitch_type="custom",
        pitch_length=105,
        pitch_width=68,
        half=True,
        pitch_color="#0f172a",
        line_color="#475569",
        line_zorder=3,
        goal_type="box",
    )
    pitch_kde.draw(ax=ax_kde)

    # Compute Gaussian spatial density
    pitch_kde.kdeplot(
        p_df["X"],
        p_df["Y"],
        ax=ax_kde,
        cmap="inferno",
        fill=True,
        levels=70,
        thresh=0.05,
        alpha=0.85,
        zorder=2,
    )

    st.pyplot(fig_kde, use_container_width=True)
    st.caption(
        "Density contours highlight high-frequency shooting zones and spatial channel bias."
    )

# 5. Tabular Data Inspection
with st.expander("📊 Inspect Event Records"):
    st.dataframe(
        p_df[
            ["Player", "Outcome", "xG", "Distance (m)", "Shot Angle (rad)"]
        ].sort_values(by="xG", ascending=False),
        use_container_width=True,
    )
