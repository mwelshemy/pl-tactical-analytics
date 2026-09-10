import streamlit as st
import pandas as pd
import requests
import re
import json
from mplsoccer import Pitch
import matplotlib.pyplot as plt

st.set_page_config(page_title="Premier League Tactical Hub", layout="wide")
st.title("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League xG & Shot Performance Hub")


# 1. Fetch Real Premier League Shot Events via Direct API Parsing
@st.cache_data(ttl=86400)
def load_pl_players():
    # Pull current/recent season league player stats
    url = "https://understat.com/league/EPL/2023"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)

    # Extract playersData JSON embedded in Understat HTML
    match = re.search(r"playersData\s*=\s*JSON\.parse\('([^']+)'\)", res.text)
    if not match:
        return pd.DataFrame()

    raw_data = match.group(1).encode("utf-8").decode("unicode_escape")
    players = json.loads(raw_data)
    return pd.DataFrame(players)


@st.cache_data(ttl=86400)
def load_player_shots(player_id):
    url = f"https://understat.com/player/{player_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)

    match = re.search(r"shotsData\s*=\s*JSON\.parse\('([^']+)'\)", res.text)
    if not match:
        return pd.DataFrame()

    raw_data = match.group(1).encode("utf-8").decode("unicode_escape")
    shots = json.loads(raw_data)
    return pd.DataFrame(shots)


# 2. UI Filters
with st.spinner("Fetching Premier League squad data..."):
    df_players = load_pl_players()

if df_players.empty:
    st.error("Unable to load player records. Please refresh the page.")
    st.stop()

teams = sorted(df_players["team_title"].unique().tolist())
selected_team = st.sidebar.selectbox("Select Premier League Club:", teams)

team_players = df_players[df_players["team_title"] == selected_team]
player_names = sorted(team_players["player_name"].unique().tolist())
selected_player = st.sidebar.selectbox("Select Player:", player_names)

# Get selected player's ID
player_row = team_players[team_players["player_name"] == selected_player].iloc[0]
player_id = player_row["id"]

# 3. Load Selected Player Shots
shots_df = load_player_shots(player_id)

if shots_df.empty:
    st.warning(f"No shot records found for {selected_player}.")
    st.stop()

# 4. Metric Cards
total_shots = len(shots_df)
total_goals = int(shots_df["result"].eq("Goal").sum())
total_xg = shots_df["xG"].astype(float).sum()
xg_per_shot = total_xg / max(total_shots, 1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Shots", total_shots)
col2.metric("Total Goals", total_goals)
col3.metric("Accumulated xG", f"{total_xg:.2f}")
col4.metric("xG / Shot Quality", f"{xg_per_shot:.2f}")

# 5. Draw Modern Half-Pitch Shot Map
pitch = Pitch(
    pitch_type="custom",
    pitch_length=105,
    pitch_width=68,
    half=True,
    pitch_color="#0f172a",
    line_color="#334155",
)
fig, ax = pitch.draw(figsize=(10, 7))

for _, shot in shots_df.iterrows():
    is_goal = shot["result"] == "Goal"
    color = "#10b981" if is_goal else "#ef4444"  # Green for goal, red for miss
    size = float(shot["xG"]) * 750 + 50

    # Understat coordinates normalized to standard pitch dimensions (105x68)
    pitch.scatter(
        float(shot["X"]) * 105,
        float(shot["Y"]) * 68,
        s=size,
        color=color,
        edgecolors="#ffffff",
        linewidth=1.2,
        alpha=0.85,
        ax=ax,
    )

st.pyplot(fig)
st.caption(
    "🟢 Green: Goal | 🔴 Red: Missed/Saved/Blocked | Bubble Size: Expected Goals (xG) Probability"
)
