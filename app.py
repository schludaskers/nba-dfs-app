import streamlit as st
import pandas as pd
import xgboost as xgb
from datetime import datetime
from nba_api.stats.endpoints import playergamelogs, scoreboardv2

# --- 1. CONFIG & HELPER FUNCTIONS ---
st.set_page_config(page_title="NBA DFS Predictor", layout="wide")

def get_teams_playing_today():
    """
    Checks the NBA schedule for 'today' and returns a list of Team IDs 
    that have a game scheduled.
    """
    # Get today's date in the format the API expects (YYYY-MM-DD)
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Pull the scoreboard for today
        board = scoreboardv2.ScoreboardV2(game_date=today_str)
        # The 'LineScore' dataset contains the Team IDs for all games on the slate
        games_df = board.line_score.get_data_frame()
        
        if games_df.empty:
            return []
            
        # Get unique Team IDs
        playing_team_ids = games_df['TEAM_ID'].unique().tolist()
        return playing_team_ids
        
    except Exception as e:
        st.error(f"Error fetching schedule: {e}")
        return []

def calculate_dk_points(row):
    """Calculates DraftKings score based on official scoring rules."""
    pts = row['PTS']
    fg3m = row['FG3M']
    reb = row['REB']
    ast = row['AST']
    stl = row['STL']
    blk = row['BLK']
    tov = row['TOV']
    
    # Base scoring
    score = (pts * 1) + (fg3m * 0.5) + (reb * 1.25) + (ast * 1.5) + (stl * 2) + (blk * 2) - (tov * 0.5)
    
    # Double-Double / Triple-Double Bonus
    stats_checking = [pts, reb, ast, stl, blk]
    double_digits = sum(1 for s in stats_checking if s >= 10)
    
    if double_digits >= 3:
        score += 3
    elif double_digits >= 2:
        score += 1.5
        
    return score

@st.cache_data
def load_data():
    """Fetches NBA game logs. Cached so it doesn't re-download on every click."""
    # Fetch data for current and previous season
    logs_24 = playergamelogs.PlayerGameLogs(season_nullable='2024-25').get_data_frames()[0]
    logs_23 = playergamelogs.PlayerGameLogs(season_nullable='2023-24').get_data_frames()[0]
    
    df = pd.concat([logs_24, logs_23])
    
    # Calculate target variable: DraftKings Points
    df['DK_PTS'] = df.apply(calculate_dk_points, axis=1)
    
    return df

def feature_engineering(df):
    """Creates rolling averages to predict future performance."""
    # Sort by player and date
    df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'])
    
    # Calculate "Last 5 Games" average for key stats
    cols_to_roll = ['MIN', 'PTS', 'REB', 'AST', 'FGA', 'DK_PTS']
    for col in cols_to_roll:
        df[f'L5_{col}'] = df.groupby('PLAYER_ID')[col].transform(lambda x: x.shift(1).rolling(window=5).mean())
    
    # Drop rows with NaN (the first 5 games of a player's season)
    df = df.dropna()
    return df

# --- 2. MAIN APP INTERFACE ---
st.title("🏀 NBA DFS Mobile Predictor")
st.write("Generating predictions for **Today's Slate** only.")

if st.button("Get Predictions for Today"):
    
    # A. Get the list of teams playing today
    with st.spinner('Checking today\'s schedule...'):
        active_teams = get_teams_playing_today()
        
    if not active_teams:
        st.warning("No games found for today (or the API returned an empty schedule).")
    else:
        st.success(f"Found {len(active_teams)} teams playing today.")

        # B. Load and Process Data
        with st.spinner('Pulling player stats & Training Model...'):
            raw_df = load_data()
            processed_df = feature_engineering(raw_df)
            
            # Features (X) and Target (y)
            features = ['L5_MIN', 'L5_PTS', 'L5_REB', 'L5_AST', 'L5_FGA', 'L5_DK_PTS']
            X = processed_df[features]
            y = processed_df['DK_PTS']
            
            # Train XGBoost Model
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
            model.fit(X, y)
            
            # C. Predict "Tonight"
            # Get the most recent stats for ALL players first
            latest_games = processed_df.groupby('PLAYER_ID').tail(1).copy()
            
            # --- THE CRITICAL FILTER STEP ---
            # Keep only players whose TEAM_ID is in the active_teams list
            today_players = latest_games[latest_games['TEAM_ID'].isin(active_teams)].copy()
            
            if today_players.empty:
                st.error("No active players found matching today's teams. (Are rosters updated?)")
            else:
                # Predict only for these players
                prediction_X = today_players[features]
                today_players['Proj_DK_PTS'] = model.predict(prediction_X)
                
                # Display nicely
                display_cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'Proj_DK_PTS', 'L5_DK_PTS', 'L5_MIN']
                st.subheader("Today's Projected Leaders")
                
                # Format the numbers nicely
                formatted_df = today_players[display_cols].sort_values(by='Proj_DK_PTS', ascending=False).head(30)
                st.dataframe(formatted_df.style.format({'Proj_DK_PTS': '{:.1f}', 'L5_DK_PTS': '{:.1f}', 'L5_MIN': '{:.1f}'}))
