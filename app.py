import streamlit as st
import pandas as pd
import xgboost as xgb
from nba_api.stats.endpoints import playergamelogs

# --- 1. CONFIG & HELPER FUNCTIONS ---
st.set_page_config(page_title="NBA DFS Predictor", layout="wide")


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
    # Fetch data for the current 2024-25 season (and previous for better training)
    # Uses Season Year format (e.g., 2024-25)
    logs_24 = playergamelogs.PlayerGameLogs(season_nullable='2024-25').get_data_frames()[0]
    logs_23 = playergamelogs.PlayerGameLogs(season_nullable='2023-24').get_data_frames()[0]

    df = pd.concat([logs_24, logs_23])

    # Calculate target variable: DraftKings Points
    df['DK_PTS'] = df.apply(calculate_dk_points, axis=1)

    return df


def feature_engineering(df):
    """Creates rolling averages to predict future performance."""
    # Sort by player and date to calculate rolling stats correctly
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
st.write("Using XGBoost to predict DraftKings scores based on Last 5 Games trends.")

if st.button("Refresh Data & Train Model"):
    with st.spinner('Pulling data from NBA API...'):
        raw_df = load_data()

    with st.spinner('Engineering features & Training...'):
        processed_df = feature_engineering(raw_df)

        # Features (X) and Target (y)
        features = ['L5_MIN', 'L5_PTS', 'L5_REB', 'L5_AST', 'L5_FGA', 'L5_DK_PTS']
        X = processed_df[features]
        y = processed_df['DK_PTS']

        # Train XGBoost Model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X, y)

        st.success("Model Trained Successfully!")

        # --- PREDICTION DEMO ---
        # Get the most recent stats for every active player to predict "Tonight"
        # In a real scenario, you'd pull tonight's schedule.
        # Here we just take every player's last known rolling stats.
        latest_games = processed_df.groupby('PLAYER_ID').tail(1)
        latest_X = latest_games[features]

        predictions = model.predict(latest_X)
        latest_games['Proj_DK_PTS'] = predictions

        # Display nicely
        display_cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'Proj_DK_PTS', 'L5_DK_PTS']
        st.subheader("Tonight's Projected Leaders")
        st.dataframe(latest_games[display_cols].sort_values(by='Proj_DK_PTS', ascending=False).head(20))