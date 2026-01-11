import streamlit as st
import pandas as pd
import xgboost as xgb
import requests
import unicodedata
import difflib
from datetime import datetime
import pytz 
from nba_api.stats.endpoints import playergamelogs, scoreboardv2, commonteamroster

# --- 1. VISUAL CONFIGURATION ---
st.set_page_config(
    page_title="CourtVision DFS",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .game-card { background-color: #1f2026; padding: 10px; border-radius: 5px; margin-bottom: 8px; border-left: 4px solid #ff4b4b; font-size: 0.9em; }
    .metric-card { background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #41444b; text-align: center; }
    h1, h2, h3 { color: #ff4b4b !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA FUNCTIONS ---

def normalize_name(name):
    """
    Aggressively cleans names to ensure 'Luka Dončić' matches 'Luka Doncic'
    """
    if not isinstance(name, str): return str(name)
    
    # 1. Unicode normalization (remove accents)
    norm = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    
    # 2. Lowercase
    norm = norm.lower()
    
    # 3. Remove common suffixes
    suffixes = [" jr.", " sr.", " ii", " iii", " iv", " jr", " sr"]
    for s in suffixes:
        if norm.endswith(s):
            norm = norm[:-len(s)]
            
    # 4. Remove punctuation
    norm = norm.replace(".", "").replace("'", "").replace("-", " ")
    
    return norm.strip()

def smart_map_names(api_names, salary_names):
    """Maps API names to Salary CSV names using fuzzy matching."""
    mapping = {}
    salary_norm_map = {normalize_name(name): name for name in salary_names}
    salary_norm_list = list(salary_norm_map.keys())
    
    for api_name in api_names:
        norm_api = normalize_name(api_name)
        if norm_api in salary_norm_map:
            mapping[api_name] = salary_norm_map[norm_api]
        else:
            matches = difflib.get_close_matches(norm_api, salary_norm_list, n=1, cutoff=0.8)
            if matches:
                mapping[api_name] = salary_norm_map[matches[0]]
                
    return mapping

@st.cache_data
def get_injury_report():
    """Scrapes CBS Sports for the latest injury report."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
        url = "https://www.cbssports.com/nba/injuries/"
        response = requests.get(url, headers=headers)
        dfs = pd.read_html(response.text)
        injury_df = pd.concat(dfs)
        
        # Normalize names
        injury_df['Player_Norm'] = injury_df['Player'].apply(normalize_name)
        
        # Filter strictly for OUT/DOUBTFUL
        out_mask = injury_df['Injury Status'].str.contains('Out|Doubtful|Injured Reserve', case=False, na=False)
        return injury_df[out_mask]['Player_Norm'].tolist()
    except Exception as e:
        print(f"Injury Scrape Error: {e}")
        return []

@st.cache_data
def get_daily_schedule():
    est = pytz.timezone('US/Eastern')
    today_str = datetime.now(est).strftime('%Y-%m-%d')
    try:
        board = scoreboardv2.ScoreboardV2(game_date=today_str)
        header = board.game_header.get_data_frame()
        linescore = board.line_score.get_data_frame()
        if header.empty: return pd.DataFrame(), []
        team_map = pd.Series(linescore.TEAM_ABBREVIATION.values, index=linescore.TEAM_ID).to_dict()
        games = []
        for _, row in header.iterrows():
            home = team_map.get(row['HOME_TEAM_ID'], '???')
            away = team_map.get(row['VISITOR_TEAM_ID'], '???')
            games.append({"Matchup": f"{away} @ {home}", "Status": row['GAME_STATUS_TEXT'].strip()})
        return pd.DataFrame(games), header['HOME_TEAM_ID'].tolist() + header['VISITOR_TEAM_ID'].tolist()
    except:
        return pd.DataFrame(), []

@st.cache_data
def get_roster_players(team_ids):
    active_ids = []
    for tid in team_ids:
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=tid).get_data_frames()[0]
            active_ids.extend(roster['PLAYER_ID'].tolist())
        except: continue
    return active_ids

def calculate_dk_points(row):
    score = (row['PTS']) + (row['FG3M'] * 0.5) + (row['REB'] * 1.25) + (row['AST'] * 1.5) + (row['STL'] * 2) + (row['BLK'] * 2) - (row['TOV'] * 0.5)
    double_digits = sum(1 for s in [row['PTS'], row['REB'], row['AST'], row['STL'], row['BLK']] if s >= 10)
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
        except: pass
    
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'])
    df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
    df['DK_PTS'] = df.apply(calculate_dk_points, axis=1)
    
    df['DAYS_REST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days - 1
    df['DAYS_REST'] = df['DAYS_REST'].fillna(3).clip(lower=0, upper=7)

    stats = ['MIN', 'PTS', 'REB', 'AST', 'FGA', 'DK_PTS']
    for col in stats:
        df[f'L5_{col}'] = df.groupby('PLAYER_ID')[col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        df[f'L10_{col}'] = df.groupby('PLAYER_ID')[col].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    return df.dropna()

# --- 3. UI LAYOUT ---

if 'schedule_df' not in st.session_state:
    st.session_state['schedule_df'] = pd.DataFrame()
if 'active_teams' not in st.session_state:
    st.session_state['active_teams'] = []

schedule_df, active_teams = get_daily_schedule()
st.session_state['schedule_df'] = schedule_df
st.session_state['active_teams'] = active_teams

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/0/03/National_Basketball_Association_logo.svg", width=80)
    st.title("CourtVision DFS")
    
    st.markdown("### 💰 Salary Data")
    salary_file = st.file_uploader("Upload DKSalaries.csv", type=['csv'])
    
    st.markdown("### 🛠️ Manual Controls")
    manual_remove = st.text_input("Force Remove Player (Partial Name)", placeholder="e.g. Embiid")
    
    st.markdown("### 📅 Today's Games")
    if not schedule_df.empty:
        for _, game in schedule_df.iterrows():
            st.markdown(f"<div class='game-card'><b>{game['Matchup']}</b><br><span style='color:#bbb'>{game['Status']}</span></div>", unsafe_allow_html=True)
    else:
        st.info("No games scheduled.")
    
    st.markdown("---")
    run_btn = st.button("🚀 Run Prediction Model", type="primary", use_container_width=True)
    show_injured = st.checkbox("Show Injured Players", value=False)

# MAIN AREA
st.title("🏀 NBA Daily Fantasy Predictor")

if run_btn:
    if not active_teams:
        st.error("❌ No games found for today!")
    else:
        status1, status2, status3 = st.columns(3)
        with status1: st.metric("Games Today", len(active_teams) // 2)
        
        active_ids = []
        with st.spinner("Fetching Rosters..."): active_ids = get_roster_players(active_teams)
        
        if not active_ids:
            st.error("⚠️ API Error: Could not fetch rosters.")
        else:
            with status2:
                with st.spinner("Checking Injuries..."):
                    # Get normalized injury list
                    injured_list_norm = get_injury_report()
                    st.metric("Injured Players Found", len(injured_list_norm))
            
            with status3: st.metric("Model Status", "Ready")
            
            with st.spinner("Running XGBoost Model..."):
                df = load_and_process_data()
                
                if not df.empty:
                    features = ['L5_DK_PTS', 'L10_DK_PTS', 'L5_MIN', 'L10_MIN', 'DAYS_REST', 'L5_FGA', 'L10_FGA']
                    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.1)
                    model.fit(df[features], df['DK_PTS'])
                    
                    latest = df.groupby('PLAYER_ID').tail(1).copy()
                    
                    # Pre-calculate normalized names
                    latest['PLAYER_NAME_NORM'] = latest['PLAYER_NAME'].apply(normalize_name)
                    
                    # 1. Filter Active Teams
                    mask_active = latest['PLAYER_ID'].isin(active_ids)
                    slate = latest[mask_active].copy()

                    # 2. Filter Injuries
                    if not show_injured:
                        # Auto-scraper filter
                        mask_auto_injury = ~slate['PLAYER_NAME_NORM'].isin(injured_list_norm)
                        slate = slate[mask_auto_injury]
                        
                        # Manual Force Remove filter
                        if manual_remove:
                            manual_norm = normalize_name(manual_remove)
                            slate = slate[~slate['PLAYER_NAME_NORM'].str.contains(manual_norm)]

                    if not slate.empty:
                        slate['Proj_DK_PTS'] = model.predict(slate[features])
                        
                        # --- SMART SALARY MERGE ---
                        if salary_file is not None:
                            try:
                                salaries = pd.read_csv(salary_file)
                                if 'Salary' in salaries.columns:
                                    # Create map
                                    api_names = slate['PLAYER_NAME'].unique().tolist()
                                    salary_names = salaries['Name'].unique().tolist()
                                    name_mapping = smart_map_names(api_names, salary_names)
                                    
                                    # Apply map
                                    slate['Matched_Name'] = slate['PLAYER_NAME'].map(name_mapping)
                                    
                                    # Merge
                                    slate = slate.merge(salaries[['Name', 'Salary']], 
                                                      left_on='Matched_Name', 
                                                      right_on='Name', 
                                                      how='left')
                                    
                                    # --- THE FIX: FILTER ZERO SALARIES ---
                                    slate = slate[slate['Salary'] > 0]
                                    slate['Value'] = slate.apply(lambda x: x['Proj_DK_PTS'] / (x['Salary']/1000), axis=1)
                                else:
                                    st.error("CSV missing 'Salary' column")
                                    slate['Salary'] = 0
                                    slate['Value'] = 0
                            except Exception as e:
                                st.warning(f"Error processing salary file: {e}")
                                slate['Salary'] = 0
                                slate['Value'] = 0
                        else:
                            slate['Salary'] = 0
                            slate['Value'] = 0

                        # --- DISPLAY ---
                        slate = slate.sort_values(by='Proj_DK_PTS', ascending=False)
                        
                        top_scorers = slate.head(3)
                        top_value = slate[slate['Proj_DK_PTS'] > 18].sort_values(by='Value', ascending=False).head(3)
                        bad_plays = slate[slate['Salary'] > 6000].sort_values(by='Value', ascending=True).head(3)

                        def draw_card(player, label_type="points"):
                            img = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player.PLAYER_ID}.png"
                            st.markdown(f"""
                            <div style="text-align:center; background-color:#262730; padding:10px; border-radius:10px; border:1px solid #444;">
                                <img src="{img}" style="width:100px; border-radius:50%;">
                                <h4>{player.PLAYER_NAME}</h4>
                                <div style="display:flex; justify-content:space-between; padding:0 20px;">
                                    <span>💰 ${int(player.Salary)}</span>
                                    <span>📊 {player.Proj_DK_PTS:.1f}</span>
                                </div>
                                <h3 style="color: {'#4CAF50' if label_type!='bad' else '#FF5252'}; margin-top:5px;">
                                    {player.Value:.1f}x Value
                                </h3>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("### 🏆 Top 3 Projected Scorers")
                        c1, c2, c3 = st.columns(3)
                        for idx, col in enumerate([c1, c2, c3]):
                            if idx < len(top_scorers): draw_card(top_scorers.iloc[idx])

                        if salary_file:
                            st.markdown("---")
                            col_v, col_b = st.columns(2)
                            with col_v:
                                st.markdown("### 💎 Top Value (Projections > 18)")
                                for i in range(len(top_value)): draw_card(top_value.iloc[i], "value")
                            with col_b:
                                st.markdown("### 🛑 Top Fades (Salary > $6k)")
                                for i in range(len(bad_plays)): draw_card(bad_plays.iloc[i], "bad")

                        st.markdown("---")
                        
                        tab1, tab2, tab3 = st.tabs(["📋 Rankings", "📊 Teams", "🛠️ Injury Debug"])
                        
                        with tab1:
                            search = st.text_input("🔍 Search", "")
                            cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'Salary', 'Proj_DK_PTS', 'Value', 'L5_DK_PTS']
                            show_df = slate[cols].copy().reset_index(drop=True)
                            if search: show_df = show_df[show_df['PLAYER_NAME'].str.contains(search, case=False)]
                            st.dataframe(show_df.style.format({'Salary': '${:.0f}', 'Proj_DK_PTS': '{:.1f}', 'Value': '{:.2f}x', 'L5_DK_PTS': '{:.1f}'}).background_gradient(subset=['Value'], cmap='RdYlGn', vmin=3, vmax=6), use_container_width=True, height=800)
                        
                        with tab2:
                            st.bar_chart(slate.groupby('TEAM_ABBREVIATION')['Proj_DK_PTS'].sum().sort_values(ascending=False))
                            
                        with tab3:
                            st.write("If you see an injured player in your rankings, look for their name in this list.")
                            st.write(f"Total Injured Players Found: {len(injured_list_norm)}")
                            st.write(injured_list_norm)

                else:
                    st.error("Error: No historical data available.")
