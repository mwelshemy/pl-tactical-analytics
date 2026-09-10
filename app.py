import streamlit as st
import soccerdata as sd
from mplsoccer import Pitch
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Premier League Shot & xG Engine", layout="wide")
st.title("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League Tactical & xG Performance Hub")


# 1. Fetch Real Premier League Understat Match Data
@st.cache_data
def get_epl_shots(season="2023"):
    understat = sd.Understat(leagues="ENG-Premier League", seasons=season)
    shots_df = understat.read_shot_events()
    return shots_df


with st.spinner("Streaming real Premier League match data..."):
    shots = get_epl_shots("2023")

# 2. Team Selection
teams = sorted(shots["team"].unique())
selected_team = st.sidebar.selectbox("Select Premier League Club:", teams)
team_shots = shots[shots["team"] == selected_team].copy()

# 3. Player Filter
players = ["All Squad"] + sorted(team_shots["player"].unique().tolist())
selected_player = st.sidebar.selectbox("Filter Player:", players)

if selected_player != "All Squad":
    display_shots = team_shots[team_shots["player"] == selected_player]
else:
    display_shots = team_shots

# 4. Advanced Metrics Overview
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Shots", len(display_shots))
col2.metric("Total Goals", int(display_shots["result"].eq("Goal").sum()))
col3.metric("Total xG", f"{display_shots['xG'].astype(float).sum():.2f}")
col4.metric(
    "xG / Shot Quality",
    f"{(display_shots['xG'].astype(float).sum() / max(len(display_shots), 1)):.2f}",
)

# 5. Render Half-Pitch Shot Map
pitch = Pitch(
    pitch_type="custom",
    pitch_length=105,
    pitch_width=68,
    half=True,
    pitch_color="#0f172a",
    line_color="#334155",
)
fig, ax = pitch.draw(figsize=(10, 7))

for _, row in display_shots.iterrows():
    is_goal = row["result"] == "Goal"
    color = (
        "#10b981" if is_goal else "#ef4444"
    )  # Emerald green for goals, crimson for misses
    size = float(row["xG"]) * 800 + 50

    # Understat coordinates normalization
    pitch.scatter(
        float(row["X"]) * 105,
        float(row["Y"]) * 68,
        s=size,
        color=color,
        edgecolors="#ffffff",
        linewidth=1.2,
        alpha=0.85,
        ax=ax,
    )

st.pyplot(fig)
st.caption(
    "🟢 Green: Goal | 🔴 Red: Missed/Saved/Blocked | Size: Expected Goals (xG) Probability"
)
