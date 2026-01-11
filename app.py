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

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .game-card {
        background-color: #1f2026;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 8px;
        border-left: 4px solid #ff4b4b;
        font-size: 0.9em;
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

@st.cache_data
def get_daily_schedule():
    today_str = datetime.now().strftime('%Y-%m-%d')
    try:
        board = scoreboardv2.ScoreboardV2(game_date=today_str)
        header = board.game_header.get_data_frame()
        linescore = board.line_score.get_data_frame()
        
        if header.empty: return pd.DataFrame(), []
            
        team_map = pd.Series(linescore.TEAM_ABBREVIATION.values, index=linescore.TEAM_ID).to_dict()
        
        games = []
        for _, row in header.iterrows():
            home_id = row['HOME_TEAM_ID']
            away_id = row['VISITOR_TEAM_ID']
            status = row['GAME_STATUS_TEXT'].strip()
            games.append({
                "Matchup": f"{team_map.get(away_id, '???')} @ {team_map.get(home_id, '???')}",
                "Status": status
            })
            
        return pd.DataFrame(games), header['HOME_TEAM_ID'].tolist() + header['VISITOR_TEAM_ID'].tolist()
    except:
        return pd.DataFrame(), []

@st.cache_data
def get_roster_players(team_ids):
    active_player_ids = []
    for tid in team_ids:
        try:
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

    # --- UPDATED ROLLING LOGIC ---
    # Added min_periods=1 so we don't drop players with <10 games history
    stats_to_roll = ['MIN', 'PTS', 'REB', 'AST', 'FGA', 'DK_PTS']
    for col in stats_to_roll:
        df[f'L5_{col}'] = df.groupby('PLAYER_ID')[col].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
        df[f'L10_{col}'] = df.groupby('PLAYER_ID')[col].transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())

    return df.dropna()

# --- 3. UI LAYOUT ---

if 'schedule_df' not in st.session_state:
    st.session_state['schedule_df'] = pd.DataFrame()
if 'active_teams' not in st.session_state:
    st.session_state['active_teams'] = []

schedule_df, active_teams = get_daily_schedule()
st.session_state['schedule_df'] = schedule_df
st.session_state['active_teams'] = active_teams

# SIDEBAR
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/0/03/National_Basketball_Association_logo.svg", width=80)
    st.title("CourtVision DFS")
    
    st.markdown("### 📅 Today's Games")
    if not schedule_df.empty:
        for _, game in schedule_df.iterrows():
            st.markdown(f"""
            <div class="game-card">
                <b>{game['Matchup']}</b><br>
                <span style="color: #bbb; font-size: 0.8em;">{game['Status']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No games scheduled today.")
    
    st.markdown("---")
    run_btn = st.button("🚀 Run Prediction Model", type="primary", use_container_width=True)
    show_injured = st.checkbox("Show Injured Players", value=False)

# MAIN AREA
st.title("🏀 NBA Daily Fantasy Predictor")

if run_btn:
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.metric("Games Today", len(active_teams) // 2)

    active_roster_player_ids = []
    if active_teams:
        with st.spinner("Fetching Live Rosters..."):
            active_roster_player_ids = get_roster_players(active_teams)

    with status_col2:
        with st.spinner("Checking Injuries..."):
            injured_players = get_injury_report()
            st.metric("Injured Players", len(injured_players))

    with status_col3:
        st.metric("Model Status", "Active", delta="Ready", delta_color="normal")

    if active_teams and active_roster_player_ids:
        with st.spinner("Crunching the numbers (XGBoost)..."):
            df = load_and_process_data()
            
            if not df.empty:
                features = ['L5_DK_PTS', 'L10_DK_PTS', 'L5_MIN', 'L10_MIN', 'DAYS_REST', 'L5_FGA', 'L10_FGA']
                X = df[features]
                y = df['DK_PTS']
                
                model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.1)
                model.fit(X, y)
                
                latest_stats = df.groupby('PLAYER_ID').tail(1).copy()
                
                active_mask = latest_stats['PLAYER_ID'].isin(active_roster_player_ids)
                injury_mask = ~latest_stats['PLAYER_NAME'].isin(injured_players) if not show_injured else True
                
                todays_slate = latest_stats[active_mask & injury_mask].copy()
                
                if not todays_slate.empty:
                    todays_slate['Proj_DK_PTS'] = model.predict(todays_slate[features])
                    todays_slate = todays_slate.sort_values(by='Proj_DK_PTS', ascending=False)
                    
                    st.markdown("### 🔥 Top 3 Projected Plays")
                    col1, col2, col3 = st.columns(3)
                    top_3 = todays_slate.head(3).itertuples()
                    
                    def player_card(col, player):
                        with col:
                            img_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player.PLAYER_ID}.png"
                            st.image(img_url, use_column_width=True)
                            st.markdown(f"<h3 style='text-align: center;'>{player.PLAYER_NAME}</h3>", unsafe_allow_html=True)
                            st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>{player.Proj_DK_PTS:.1f} PTS</h2>", unsafe_allow_html=True)

                    players = list(top_3)
                    if len(players) >= 1: player_card(col1, players[0])
                    if len(players) >= 2: player_card(col2, players[1])
                    if len(players) >= 3: player_card(col3, players[2])

                    st.markdown("---")

                    tab1, tab2 = st.tabs(["📋 Full Rankings", "📊 Team Breakdown"])
                    
                    with tab1:
                        # --- UPDATED TABLE ---
                        # 1. Added Search/Filter
                        search = st.text_input("🔍 Search Player or Team", "")
                        
                        display_cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'Proj_DK_PTS', 'L5_DK_PTS', 'DAYS_REST']
                        display_df = todays_slate[display_cols].copy()
                        
                        if search:
                            display_df = display_df[
                                display_df['PLAYER_NAME'].str.contains(search, case=False) | 
                                display_df['TEAM_ABBREVIATION'].str.contains(search, case=False)
                            ]
                        
                        # 2. REMOVED .head(50) LIMIT -> Shows all rows now
                        try:
                            styled_df = display_df.style\
                                .format({'Proj_DK_PTS': '{:.1f}', 'L5_DK_PTS': '{:.1f}', 'DAYS_REST': '{:.0f}'})\
                                .background_gradient(subset=['Proj_DK_PTS'], cmap='Greens')
                        except:
                            styled_df = display_df.style\
                                .format({'Proj_DK_PTS': '{:.1f}', 'L5_DK_PTS': '{:.1f}', 'DAYS_REST': '{:.0f}'})
                        
                        st.dataframe(styled_df, use_container_width=True, height=800)
                        
                    with tab2:
                        team_proj = todays_slate.groupby('TEAM_ABBREVIATION')['Proj_DK_PTS'].sum().sort_values(ascending=False)
                        st.bar_chart(team_proj)
                else:
                    st.warning("No players found matching today's rosters.")
