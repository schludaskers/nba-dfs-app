import streamlit as st
import pandas as pd
import xgboost as xgb
from datetime import datetime
from nba_api.stats.endpoints import playergamelogs, scoreboardv2, commonteamroster

# --- 1. VISUAL CONFIGURATION ---
st.set_page_config(
    page_title="CourtVision DFS",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Dark Mode" Sports App Vibe
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #41444b;
        text-align: center;
    }
    h1, h2, h3 {
        color: #ff4b4b !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA FUNCTIONS ---

@st.cache_data
def get_injury_report():
    try:
        url = "https://www.cbssports.com/nba/injuries/"
        dfs = pd.read_html(url)
        injury_df = pd.concat(dfs)
        out_players = injury_df[injury_df['Injury Status'].str.contains('Out', case=False, na=False)]
        return out_players['Player'].tolist()
    except:
        return []

def get_teams_playing_today():
    """Returns a list of Team IDs for teams playing today."""
    today_str = datetime.now().strftime('%Y-%m-%d')
    try:
        board = scoreboardv2.ScoreboardV2(game_date=today_str)
        games_df = board.line_score.get_data_frame()
        if games_df.empty: return []
        return games_df['TEAM_ID'].unique().tolist()
    except:
        return []

@st.cache_data
def get_roster_players(team_ids):
    """
    Fetches the OFFICIAL current roster for every team playing today.
    This fixes the issue of missing traded players or bench players.
    """
    active_player_ids = []
    
    # We loop through every team playing today and get their live roster
    for tid in team_ids:
        try:
            # commonteamroster endpoint gets the current roster
            roster = commonteamroster.CommonTeamRoster(team_id=tid).get_data_frames()[0]
            active_player_ids.extend(roster['PLAYER_ID'].tolist())
        except:
            continue
            
    return active_player_ids

def calculate_dk_points(row):
    pts = row['PTS']
    fg3m = row['FG3M']
    reb = row['REB']
    ast = row['AST']
    stl = row['STL']
    blk = row['BLK']
    tov = row['TOV']
    score = (pts * 1) + (fg3m * 0.5) + (reb * 1.25) + (ast * 1.5) + (stl * 2) + (blk * 2) - (tov * 0.5)
    double_digits = sum(1 for s in [pts, reb, ast, stl, blk] if s >= 10)
    if double_digits >= 3: score += 3
    elif double_digits >= 2: score += 1.5
    return score

@st.cache_data
def load_and_process_data():
    seasons = ['2024-25', '2025-26']
    dfs = []
    for season in seasons:
        try:
            logs = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
            dfs.append(logs)
        except:
            pass 
    
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'])
    df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
    df['DK_PTS'] = df.apply(calculate_dk_points, axis=1)
    
    df['DAYS_REST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days - 1
    df['DAYS_REST'] = df['DAYS_REST'].fillna(3).clip(lower=0, upper=7)

    stats_to_roll = ['MIN', 'PTS', 'REB', 'AST', 'FGA', 'DK_PTS']
    for col in stats_to_roll:
        df[f'L5_{col}'] = df.groupby('PLAYER_ID')[col].transform(lambda x: x.shift(1).rolling(window=5).mean())
        df[f'L10_{col}'] = df.groupby('PLAYER_ID')[col].transform(lambda x: x.shift(1).rolling(window=10).mean())

    return df.dropna()

# --- 3. UI LAYOUT ---

# SIDEBAR
with st.sidebar:
    # Use Wikimedia's public URL for the NBA logo to avoid access errors
    st.image("https://upload.wikimedia.org/wikipedia/en/0/03/National_Basketball_Association_logo.svg", width=100)
    st.title("CourtVision DFS")
    st.markdown("---")
    
    run_btn = st.button("🚀 Run Prediction Model", type="primary", use_container_width=True)
    
    st.markdown("### ⚙️ Settings")
    show_injured = st.checkbox("Show Injured Players", value=False)
    
    st.markdown("---")
    st.info("Data Sources: NBA.com API & CBS Sports (Injuries)")

# MAIN AREA
st.title("🏀 NBA Daily Fantasy Predictor")
st.markdown("#### AI-Powered Projections for DraftKings")

if run_btn:
    # STATUS INDICATORS
    status_col1, status_col2, status_col3 = st.columns(3)
    
    # 1. Get Teams Playing Today
    with status_col1:
        with st.spinner("Checking Schedule..."):
            active_teams = get_teams_playing_today()
            if active_teams:
                st.metric("Games Today", len(active_teams) // 2)
            else:
                st.error("No Games Today")

    # 2. Get Official Rosters for those teams (THE FIX)
    active_roster_player_ids = []
    if active_teams:
        with st.spinner("Fetching Live Rosters..."):
            active_roster_player_ids = get_roster_players(active_teams)

    # 3. Get Injuries
    with status_col2:
        with st.spinner("Checking Injuries..."):
            injured_players = get_injury_report()
            st.metric("Injured Players", len(injured_players))

    with status_col3:
        st.metric("Model Status", "Active", delta="Ready", delta_color="normal")

    # MODEL EXECUTION
    if active_teams and active_roster_player_ids:
        with st.spinner("Crunching the numbers (XGBoost)..."):
            df = load_and_process_data()
            
            if not df.empty:
                # Features & Training
                features = ['L5_DK_PTS', 'L10_DK_PTS', 'L5_MIN', 'L10_MIN', 'DAYS_REST', 'L5_FGA', 'L10_FGA']
                X = df[features]
                y = df['DK_PTS']
                
                model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.1)
                model.fit(X, y)
                
                # Predict
                # Take the LAST known stats for every player in history
                latest_stats = df.groupby('PLAYER_ID').tail(1).copy()
                
                # --- FILTERING FIX ---
                # Instead of filtering by "Team in Last Game", we filter by "Is Player ID in Today's Roster?"
                # This catches players who were traded or haven't played recently but are active today.
                active_mask = latest_stats['PLAYER_ID'].isin(active_roster_player_ids)
                
                # Filter Injuries
                injury_mask = ~latest_stats['PLAYER_NAME'].isin(injured_players) if not show_injured else True
                
                todays_slate = latest_stats[active_mask & injury_mask].copy()
                
                if not todays_slate.empty:
                    todays_slate['Proj_DK_PTS'] = model.predict(todays_slate[features])
                    todays_slate = todays_slate.sort_values(by='Proj_DK_PTS', ascending=False)
                    
                    # --- DISPLAY: TOP 3 HERO SECTION ---
                    st.markdown("### 🔥 Top 3 Projected Plays")
                    col1, col2, col3 = st.columns(3)
                    
                    top_3 = todays_slate.head(3).itertuples()
                    
                    def player_card(col, player):
                        with col:
                            img_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player.PLAYER_ID}.png"
                            st.image(img_url, use_column_width=True)
                            st.markdown(f"<h3 style='text-align: center;'>{player.PLAYER_NAME}</h3>", unsafe_allow_html=True)
                            st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>{player.Proj_DK_PTS:.1f} PTS</h2>", unsafe_allow_html=True)
                            st.caption(f"Last 5 Avg: {player.L5_DK_PTS:.1f}")

                    players = list(top_3)
                    if len(players) >= 1: player_card(col1, players[0])
                    if len(players) >= 2: player_card(col2, players[1])
                    if len(players) >= 3: player_card(col3, players[2])

                    st.markdown("---")

                    # --- TABS FOR DETAILED DATA ---
                    tab1, tab2 = st.tabs(["📋 Full Rankings", "📊 Team Breakdown"])
                    
                    with tab1:
                        display_cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'Proj_DK_PTS', 'L5_DK_PTS', 'DAYS_REST']
                        
                        # Apply style with error handling (in case matplotlib is missing)
                        try:
                            styled_df = todays_slate[display_cols].head(50).style\
                                .format({'Proj_DK_PTS': '{:.1f}', 'L5_DK_PTS': '{:.1f}', 'DAYS_REST': '{:.0f}'})\
                                .background_gradient(subset=['Proj_DK_PTS'], cmap='Greens')
                        except:
                            # Fallback if matplotlib isn't installed
                            styled_df = todays_slate[display_cols].head(50).style\
                                .format({'Proj_DK_PTS': '{:.1f}', 'L5_DK_PTS': '{:.1f}', 'DAYS_REST': '{:.0f}'})

                        st.dataframe(styled_df, use_container_width=True, height=600)
                        
                    with tab2:
                        # Chart
                        team_proj = todays_slate.groupby('TEAM_ABBREVIATION')['Proj_DK_PTS'].sum().sort_values(ascending=False)
                        st.bar_chart(team_proj)
                        st.caption("Which teams are expected to score the most fantasy points today?")

                else:
                    st.warning("No players found matching today's rosters. (Check if season is active).")
    else:
        if not active_teams:
            st.error("Cannot run model: No games found for today.")
