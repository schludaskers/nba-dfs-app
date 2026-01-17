import streamlit as st
import pandas as pd
import xgboost as xgb
import requests
import unicodedata
import difflib
import io
import numpy as np
from datetime import datetime
import pytz 
from nba_api.stats.endpoints import scoreboardv2, commonteamroster

# --- 1. CONFIGURATION ---
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
    h1, h2, h3 { color: #ff4b4b !important; }
    
    /* UNIFIED CARD STYLING */
    div.player-card {
        background-color: #262730; 
        padding: 15px; 
        border-radius: 12px; 
        border: 1px solid #444; 
        margin-bottom: 10px;
        text-align: center;
        position: relative;
        height: 320px; /* Fixed Height for Alignment */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s;
    }
    div.player-card:hover {
        transform: scale(1.02);
        border-color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA FUNCTIONS ---

def get_season_string(date_obj):
    year = date_obj.year
    month = date_obj.month
    if month >= 10: start_year = year
    else: start_year = year - 1
    end_year_short = (start_year + 1) % 100
    return f"{start_year}-{end_year_short:02d}"

def normalize_name(name):
    if not isinstance(name, str): return str(name)
    norm = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    norm = norm.lower().strip()
    norm = norm.replace(".", "").replace("'", "").replace("-", " ")
    return norm.strip()

def smart_map_names(api_names, salary_names):
    mapping = {}
    salary_norm_map = {normalize_name(name): name for name in salary_names}
    salary_norm_list = list(salary_norm_map.keys())
    for api_name in api_names:
        norm_api = normalize_name(api_name)
        if norm_api in salary_norm_map:
            mapping[api_name] = salary_norm_map[norm_api]
        else:
            candidates = [n for n in salary_norm_list if n.startswith(norm_api[0])]
            matches = difflib.get_close_matches(norm_api, candidates, n=1, cutoff=0.90)
            if matches: mapping[api_name] = salary_norm_map[matches[0]]
    return mapping

@st.cache_data
def get_injury_report():
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = "https://www.espn.com/nba/injuries"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(io.StringIO(response.text))
        all_injuries = []
        for df in dfs:
            if len(df.columns) >= 3:
                sample_val = str(df.iloc[0, 1]).upper()
                if sample_val in ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F']:
                    row_vals = [str(x).lower() for x in df.iloc[0].values]
                    status_idx = -1
                    for i, val in enumerate(row_vals):
                        if any(k in val for k in ['out', 'day-to-day', 'questionable', 'doubtful']):
                            status_idx = i
                            break
                    target_idx = status_idx if status_idx != -1 else (3 if len(df.columns) > 3 else 2)
                    clean_df = df.iloc[:, [0, target_idx]].copy()
                    clean_df.columns = ['Player', 'Injury Status']
                    all_injuries.append(clean_df)
                    continue

            df.columns = [str(c).upper() for c in df.columns]
            name_col = next((c for c in df.columns if 'NAME' in c or 'PLAYER' in c), None)
            status_col = next((c for c in df.columns if 'STATUS' in c), None)
            if name_col and status_col:
                clean_df = df[[name_col, status_col]].copy()
                clean_df.columns = ['Player', 'Injury Status']
                clean_df = clean_df[clean_df['Player'] != 'NAME']
                all_injuries.append(clean_df)
        
        if all_injuries:
            combined = pd.concat(all_injuries)
            combined['Player_Norm'] = combined['Player'].apply(normalize_name)
            return combined[['Player_Norm', 'Injury Status']]
    except: pass
    return pd.DataFrame(columns=['Player_Norm', 'Injury Status'])

@st.cache_data
def get_daily_schedule(selected_date):
    date_str = selected_date.strftime('%Y-%m-%d')
    try:
        board = scoreboardv2.ScoreboardV2(game_date=date_str, timeout=30)
        header = board.game_header.get_data_frame()
        linescore = board.line_score.get_data_frame()
        if header.empty: return pd.DataFrame(), [], {}
        
        team_map = pd.Series(linescore.TEAM_ABBREVIATION.values, index=linescore.TEAM_ID).to_dict()
        opponent_map = {}
        games = []
        active_teams = []
        
        for _, row in header.iterrows():
            home_id = str(row['HOME_TEAM_ID'])
            away_id = str(row['VISITOR_TEAM_ID'])
            opponent_map[home_id] = away_id
            opponent_map[away_id] = home_id
            active_teams.extend([home_id, away_id])
            home_name = team_map.get(int(home_id), '???')
            away_name = team_map.get(int(away_id), '???')
            games.append({"Matchup": f"{away_name} @ {home_name}", "Status": row['GAME_STATUS_TEXT']})
            
        return pd.DataFrame(games), active_teams, opponent_map
    except: return pd.DataFrame(), [], {}

@st.cache_data
def get_roster_data(team_ids):
    all_rosters = []
    for tid in team_ids:
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=tid, timeout=30).get_data_frames()[0]
            roster['PLAYER_ID'] = roster['PLAYER_ID'].astype(str)
            all_rosters.append(roster[['PLAYER_ID', 'POSITION']])
        except: continue
    if all_rosters: return pd.concat(all_rosters)
    return pd.DataFrame(columns=['PLAYER_ID', 'POSITION'])

def get_usage_and_defense():
    try:
        usage_df = pd.read_csv('nba_usage_2026.csv')
        usage_df['PLAYER_ID'] = usage_df['PLAYER_ID'].astype(str)
        if 'PLAYER_NAME' in usage_df.columns:
            usage_df = usage_df.rename(columns={'PLAYER_NAME': 'Usage_Name'})

        defense_df = pd.read_csv('nba_defense_2026.csv')
        defense_df['TEAM_ID'] = defense_df['TEAM_ID'].astype(str) 
        return usage_df, defense_df
    except: return pd.DataFrame(), pd.DataFrame()

def calculate_dk_points(row):
    score = (row['PTS']) + (row['FG3M'] * 0.5) + (row['REB'] * 1.25) + (row['AST'] * 1.5) + (row['STL'] * 2) + (row['BLK'] * 2) - (row['TOV'] * 0.5)
    double_digits = sum(1 for s in [row['PTS'], row['REB'], row['AST'], row['STL'], row['BLK']] if s >= 10)
    if double_digits >= 3: score += 3
    elif double_digits >= 2: score += 1.5
    return score

@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv('nba_gamelogs_2026.csv')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    except: return pd.DataFrame()
        
    if df.empty: return pd.DataFrame()

    df['PLAYER_ID'] = df['PLAYER_ID'].astype(str)
    df['TEAM_ID'] = df['TEAM_ID'].astype(str)
    df['OPPONENT'] = df['MATCHUP'].apply(lambda x: x.replace('@', 'vs.').split(' vs. ')[-1])

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

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/0/03/National_Basketball_Association_logo.svg", width=80)
    st.title("CourtVision DFS")
    est = pytz.timezone('US/Eastern')
    selected_date = st.date_input("📅 Game Date", datetime.now(est))
    current_season_str = get_season_string(selected_date)
    st.caption(f"Season: {current_season_str}")
    
    st.markdown("### 💰 Salary Data")
    salary_file = st.file_uploader("Upload DKSalaries.csv", type=['csv'])
    
    schedule_df, active_teams, opponent_map = get_daily_schedule(selected_date)
    st.markdown("### 📅 Games Schedule")
    if not schedule_df.empty:
        for _, game in schedule_df.iterrows():
            st.markdown(f"<div class='game-card'><b>{game['Matchup']}</b><br><span style='color:#bbb'>{game['Status']}</span></div>", unsafe_allow_html=True)
    else: st.info("No games scheduled.")
    
    st.markdown("---")
    run_btn = st.button("🚀 Run Prediction Model", type="primary", use_container_width=True)
    show_injured = st.checkbox("Show Injured Players", value=False)

st.title("🏀 NBA Daily Fantasy Predictor")

if run_btn:
    if not active_teams:
        st.error(f"❌ No games found for {selected_date.strftime('%Y-%m-%d')}!")
    else:
        status1, status2, status3 = st.columns(3)
        with status1: st.metric("Games On Slate", len(active_teams) // 2)
        
        roster_df = pd.DataFrame()
        with st.spinner("Fetching Rosters..."): 
            roster_df = get_roster_data(active_teams)
            active_ids = roster_df['PLAYER_ID'].tolist() if not roster_df.empty else []
        
        if not active_ids: st.error("⚠️ API Error: Could not fetch rosters.")
        else:
            with status2:
                with st.spinner("Checking Injuries..."):
                    injury_df = get_injury_report()
                    status_map = {}
                    injured_list_norm = []
                    if not injury_df.empty:
                        status_map = dict(zip(injury_df['Player_Norm'], injury_df['Injury Status']))
                        exclude_mask = injury_df['Injury Status'].str.contains('Out|Doubtful|Injured Reserve', case=False, na=False)
                        injured_list_norm = injury_df[exclude_mask]['Player_Norm'].tolist()
                    st.metric("Injured Players", len(injured_list_norm))
            
            with status3: st.metric("Model Status", "XGBoost Active")
            
            usage_df, defense_df = get_usage_and_defense()

            with st.spinner("Loading Stats & Running Model..."):
                df = load_and_process_data()
                
                if not df.empty:
                    # H2H Logic
                    h2h_df = df.sort_values('GAME_DATE').groupby(['PLAYER_ID', 'OPPONENT'])['DK_PTS'].apply(lambda x: x.tail(5).mean()).reset_index()
                    h2h_df.rename(columns={'DK_PTS': 'H2H_Avg'}, inplace=True)
                    
                    team_id_to_abbrev = df[['TEAM_ID', 'TEAM_ABBREVIATION']].drop_duplicates().set_index('TEAM_ID')['TEAM_ABBREVIATION'].to_dict()

                    # Train
                    features = ['L5_DK_PTS', 'L10_DK_PTS', 'L5_MIN', 'L10_MIN', 'DAYS_REST', 'L5_FGA', 'L10_FGA']
                    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.1)
                    model.fit(df[features], df['DK_PTS'])
                    
                    # Slate
                    latest = df.groupby('PLAYER_ID').tail(1).copy()
                    latest['PLAYER_NAME_NORM'] = latest['PLAYER_NAME'].apply(normalize_name)
                    slate = latest[latest['PLAYER_ID'].isin(active_ids)].copy()
                    
                    # Merges
                    slate['OPP_TEAM_ID'] = slate['TEAM_ID'].map(opponent_map)
                    slate['OPP_ABBREV'] = slate['OPP_TEAM_ID'].map(team_id_to_abbrev)
                    slate = slate.merge(h2h_df, left_on=['PLAYER_ID', 'OPP_ABBREV'], right_on=['PLAYER_ID', 'OPPONENT'], how='left')
                    slate['H2H_Avg'] = slate['H2H_Avg'].fillna(slate['L10_DK_PTS'])

                    if not roster_df.empty: slate = slate.merge(roster_df, on='PLAYER_ID', how='left')
                    if not usage_df.empty:
                        slate = slate.merge(usage_df[['PLAYER_ID', 'USG_PCT']], on='PLAYER_ID', how='left')
                        slate['USG_PCT'] = slate['USG_PCT'].fillna(0.20)
                    else: slate['USG_PCT'] = 0.20

                    if not defense_df.empty:
                        slate = slate.merge(defense_df, left_on='OPP_TEAM_ID', right_on='TEAM_ID', how='left')
                        slate['Def_Rank'] = slate['Def_Rank'].fillna(15)
                    else: slate['Def_Rank'] = 15

                    if not show_injured: slate = slate[~slate['PLAYER_NAME_NORM'].isin(injured_list_norm)]

                    if not slate.empty:
                        slate['Base_Proj'] = model.predict(slate[features])
                        slate['Proj_DK_PTS'] = (slate['Base_Proj'] * 0.85) + (slate['H2H_Avg'] * 0.15)
                        slate['Injury Status'] = slate['PLAYER_NAME_NORM'].map(status_map).fillna("")

                        if salary_file is not None:
                            try:
                                salaries = pd.read_csv(salary_file)
                                if 'Name' not in salaries.columns and 'Player Name' in salaries.columns:
                                    salaries = salaries.rename(columns={'Player Name': 'Name'})
                                if 'Name' in salaries.columns:
                                    api_names = slate['PLAYER_NAME'].unique().tolist()
                                    salary_names = salaries['Name'].unique().tolist()
                                    name_mapping = smart_map_names(api_names, salary_names)
                                    slate['Matched_Name'] = slate['PLAYER_NAME'].map(name_mapping)
                                    slate = slate.merge(salaries[['Name', 'Salary']], left_on='Matched_Name', right_on='Name', how='left')
                                    slate['Salary'] = pd.to_numeric(slate['Salary'], errors='coerce').fillna(0)
                                    slate = slate[slate['Salary'] > 0]
                                    slate['Value'] = slate.apply(lambda x: x['Proj_DK_PTS'] / (x['Salary']/1000) if x['Salary'] > 0 else 0, axis=1)
                                else: slate['Salary'] = 0; slate['Value'] = 0
                            except: slate['Salary'] = 0; slate['Value'] = 0
                        else: slate['Salary'] = 0; slate['Value'] = 0

                        # Sections
                        slate = slate.sort_values(by='Proj_DK_PTS', ascending=False)
                        top_scorers = slate.head(3)
                        top_value = slate[slate['Proj_DK_PTS'] > 18].sort_values(by='Value', ascending=False).head(3)
                        top_fades = slate[slate['Salary'] >= 6000].sort_values(by='Value', ascending=True).head(3)

                        # --- ROBUST CARD DRAWING ---
                        def draw_card(player, label=None):
                            # Ensure inputs are safe
                            name = player.get('PLAYER_NAME', 'Unknown')
                            pid = player.get('PLAYER_ID', '')
                            pos = player.get('POSITION', 'UNK')
                            if pd.isna(pos): pos = 'UNK'
                            
                            status = player.get('Injury Status', '')
                            if pd.isna(status): status = ''
                            
                            # Safely handle Numbers (Convert NaNs to 0)
                            usg = player.get('USG_PCT', 0.2)
                            if pd.isna(usg): usg = 0.2
                            usg_val = usg * 100
                            
                            sal = player.get('Salary', 0)
                            if pd.isna(sal): sal = 0
                            
                            proj = player.get('Proj_DK_PTS', 0)
                            if pd.isna(proj): proj = 0
                            
                            val = player.get('Value', 0)
                            if pd.isna(val): val = 0
                            
                            opp = player.get('OPP_ABBREV', 'OPP')
                            if pd.isna(opp): opp = 'OPP'
                            
                            h2h = player.get('H2H_Avg', 0)
                            if pd.isna(h2h): h2h = 0
                            
                            val_color = "#4CAF50" if val >= 5 else "#FFC107" if val >= 4 else "#F44336"
                            img = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"
                            label_html = f"<div style='position:absolute; top:5px; right:5px; background:#333; padding:2px 6px; border-radius:4px; font-size:0.7em; color:#ddd;'>{label}</div>" if label else ""

                            st.markdown(
                                f"""
                                <div class="player-card">
                                    {label_html}
                                    <div style="display:flex; justify-content:center;">
                                        <img src="{img}" style="width:80px; height:80px; border-radius:50%; border: 2px solid #333; object-fit: cover;" onerror="this.onerror=null; this.src='https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png'">
                                    </div>
                                    <div>
                                        <div style="font-weight:bold; font-size:1.1em; margin-bottom:2px;">{name}</div>
                                        <div style="font-size:0.8em; color:#bbb; margin-bottom:5px;">{pos}</div>
                                        <div style="color:#FFC107; font-size:0.8em; min-height:15px; margin-bottom:5px;">{status}</div>
                                    </div>
                                    <div style="background:#333; border-radius:4px; padding:4px; margin-bottom:8px; font-size:0.8em;">
                                        <span style="color:#aaa;">Vs {opp}:</span> <span style="color:#fff; font-weight:bold;">{h2h:.1f}</span>
                                    </div>
                                    <div>
                                        <div style="display:flex; justify-content:space-between; background:#1e1e24; padding:8px 12px; border-radius:6px;">
                                            <span style="color:#ddd;">💰 ${int(sal)}</span>
                                            <span style="color:#fff; font-weight:bold;">📊 {proj:.1f}</span>
                                        </div>
                                        <div style="color: {val_color}; font-size:1.4em; font-weight:800; margin-top:8px;">
                                            {val:.1f}x <span style="font-size:0.6em; font-weight:normal; color:#aaa;">VAL</span>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True
                            )

                        st.markdown("### 🏆 Top 3 Projected Scorers")
                        c1, c2, c3 = st.columns(3)
                        for idx, col in enumerate([c1, c2, c3]):
                            if idx < len(top_scorers): with col: draw_card(top_scorers.iloc[idx], "🔥 Top Scorer")

                        st.markdown("### 💰 Top 3 Value Plays (Best ROI)")
                        c1, c2, c3 = st.columns(3)
                        for idx, col in enumerate([c1, c2, c3]):
                            if idx < len(top_value): with col: draw_card(top_value.iloc[idx], "💎 Value")
                                
                        st.markdown("### 📉 Top 3 Fades (Avoid)")
                        c1, c2, c3 = st.columns(3)
                        for idx, col in enumerate([c1, c2, c3]):
                            if idx < len(top_fades): with col: draw_card(top_fades.iloc[idx], "🛑 Fade")

                        st.markdown("---")
                        tab1, tab2 = st.tabs(["📋 Rankings", "🛠️ Debug"])
                        
                        with tab1:
                            cols = ['PLAYER_NAME', 'POSITION', 'TEAM_ABBREVIATION', 'OPP_ABBREV', 'Injury Status', 'Def_Rank', 'H2H_Avg', 'Salary', 'Proj_DK_PTS', 'Value']
                            valid_cols = [c for c in cols if c in slate.columns]
                            st.dataframe(slate[valid_cols].sort_values('Value', ascending=False)
                                         .style.format({'Salary': '${:.0f}', 'Proj_DK_PTS': '{:.1f}', 'H2H_Avg': '{:.1f}', 'Value': '{:.2f}x', 'Def_Rank': '{:.0f}'})
                                         .background_gradient(subset=['Value'], cmap='RdYlGn', vmin=3, vmax=6))
                        
                        with tab2:
                            st.write(f"Usage Rows: {len(usage_df)}")
                            st.write(f"Defense Rows: {len(defense_df)}")
                            st.write(slate.head())
                else: st.error("❌ Stats file empty or not found.")
