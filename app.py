import streamlit as st
import pandas as pd
import xgboost as xgb
from datetime import datetime
from nba_api.stats.endpoints import playergamelogs, scoreboardv2

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Pro NBA DFS Predictor", layout="wide")

# --- 2. DATA FUNCTIONS ---

@st.cache_data
def get_injury_report():
    """
    Scrapes a public injury report to get a list of players who are 'Out'.
    This is a 'hack' because official Injury APIs are expensive.
    """
    try:
        # CBS Sports has a clean table structure that pandas can read easily
        url = "https://www.cbssports.com/nba/injuries/"
        dfs = pd.read_html(url)
        injury_df = pd.concat(dfs)
        
        # Filter for players who are definitely OUT
        # Keywords: 'Out', 'Expected to be out', etc.
        out_players = injury_df[injury_df['Injury Status'].str.contains('Out', case=False, na=False)]
        return out_players['Player'].tolist()
    except Exception as e:
        st.error(f"Could not load injury report: {e}")
        return []

def get_teams_playing_today():
    """Gets the list of teams playing on today's slate."""
    today_str = datetime.now().strftime('%Y-%m-%d')
    try:
        board = scoreboardv2.ScoreboardV2(game_date=today_str)
        games_df = board.line_score.get_data_frame()
        if games_df.empty: return []
        return games_df['TEAM_ID'].unique().tolist()
    except:
        return []

def calculate_dk_points(row):
    """DraftKings Scoring Formula"""
    pts = row['PTS']
    fg3m = row['FG3M']
    reb = row['REB']
    ast = row['AST']
    stl = row['STL']
    blk = row['BLK']
    tov = row['TOV']
    
    score = (pts * 1) + (fg3m * 0.5) + (reb * 1.25) + (ast * 1.5) + (stl * 2) + (blk * 2) - (tov * 0.5)
    
    # Bonuses
    double_digits = sum(1 for s in [pts, reb, ast, stl, blk] if s >= 10)
    if double_digits >= 3: score += 3
    elif double_digits >= 2: score += 1.5
    return score

@st.cache_data
def load_and_process_data():
    """
    1. Pulls 2 seasons of logs.
    2. Calculates Days of Rest.
    3. Calculates L5/L10 Rolling Stats.
    """
    # 1. Pull Data
    seasons = ['2024-25', '2025-26']
    dfs = []
    for season in seasons:
        try:
            logs = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
            dfs.append(logs)
        except:
            pass # Skip if season hasn't started or fails
            
    df = pd.concat(dfs)
    
    # 2. Basic Cleanup
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'])
    
    # 3. Calculate Target (DK Points)
    df['DK_PTS'] = df.apply(calculate_dk_points, axis=1)
    
    # 4. Feature Engineering: Days of Rest
    # Calculate difference in days between current game and previous game for each player
    df['DAYS_REST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days - 1
    df['DAYS_REST'] = df['DAYS_REST'].fillna(3) # Default to 3 days rest for first game of season
    # Cap rest at 7 days (longer breaks don't add linear value)
    df['DAYS_REST'] = df['DAYS_REST'].clip(lower=0, upper=7)

    # 5. Feature Engineering: Rolling Stats (L5 and L10)
    stats_to_roll = ['MIN', 'PTS', 'REB', 'AST', 'FGA', 'DK_PTS', 'USG_PCT'] # USG_PCT isn't in base logs, we use FGA as proxy for usage
    
    for col in stats_to_roll:
        # Last 5 Games
        df[f'L5_{col}'] = df.groupby('PLAYER_ID')[col].transform(lambda x: x.shift(1).rolling(window=5).mean())
        # Last 10 Games
        df[f'L10_{col}'] = df.groupby('PLAYER_ID')[col].transform(lambda x: x.shift(1).rolling(window=10).mean())

    # Drop early season games where we don't have enough history for rolling stats
    df = df.dropna()
    
    return df

# --- 3. MAIN APP ---
st.title("🏀 Pro NBA DFS Model")

if st.button("Run Prediction Model"):
    
    # 1. Get Context (Teams & Injuries)
    with st.spinner("Checking Schedule & Injury Reports..."):
        active_teams = get_teams_playing_today()
        injured_players = get_injury_report()
        st.write(f"**Injured Players Found:** {len(injured_players)} (filtered out)")

    if not active_teams:
        st.error("No games scheduled for today.")
    else:
        # 2. Load Data
        with st.spinner("Processing stats from 2024-26 seasons..."):
            df = load_and_process_data()
            
            # Define Features
            features = [
                'L5_DK_PTS', 'L10_DK_PTS',  # Recent Form
                'L5_MIN', 'L10_MIN',        # Minutes Security
                'DAYS_REST',                # Fatigue
                'L5_FGA', 'L10_FGA'         # Volume/Usage proxy
            ]
            
            X = df[features]
            y = df['DK_PTS']
            
            # 3. Train Model
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.1)
            model.fit(X, y)
            
            # 4. Predict for Tonight
            # Take the LAST known stats for every player
            latest_stats = df.groupby('PLAYER_ID').tail(1).copy()
            
            # Filter: Must be playing today AND Not Injured
            active_mask = latest_stats['TEAM_ID'].isin(active_teams)
            injury_mask = ~latest_stats['PLAYER_NAME'].isin(injured_players)
            
            todays_slate = latest_stats[active_mask & injury_mask].copy()
            
            if todays_slate.empty:
                st.warning("No active players found. (Check if season is active).")
            else:
                # Predict
                todays_slate['Proj_DK_PTS'] = model.predict(todays_slate[features])
                
                # Show Top 50
                cols = ['PLAYER_NAME', 'Proj_DK_PTS', 'L5_DK_PTS', 'DAYS_REST']
                st.subheader("🔥 Top Predicted Plays")
                st.dataframe(todays_slate[cols].sort_values(by='Proj_DK_PTS', ascending=False).head(50))
