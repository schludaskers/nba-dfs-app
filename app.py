import streamlit as st
import pandas as pd
import xgboost as xgb
import requests
import unicodedata
import difflib
import io
import textwrap
from datetime import datetime, timedelta
import pytz 
from nba_api.stats.endpoints import playergamelogs, scoreboardv2, commonteamroster, leaguedashplayerstats, leaguedashteamstats

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
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA FUNCTIONS ---

def get_season_string(date_obj):
    """Calculates the NBA Season string (e.g., '2025-26') from a date."""
    year = date_obj.year
    month = date_obj.month
    # If it's late in the year (Oct-Dec), the season starts this year.
    # If it's early (Jan-June), the season started last year.
    if month >= 10:
        start_year = year
    else:
        start_year = year - 1
    
    end_year_short = (start_year + 1) % 100
    return f"{start_year}-{end_year_short:02d}"

def normalize_name(name):
    if not isinstance(name, str): return str(name)
    norm = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    norm = norm.lower().strip()
    suffixes = [" jr.", " sr.", " ii", " iii", " iv", " jr", " sr"]
    for s in suffixes:
        if norm.endswith(s):
            norm = norm[:-len(s)]
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
            if matches:
                mapping[api_name] = salary_norm_map[matches[0]]
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
            df.columns = [str(c).upper() for c in df.columns]
            name_col = next((c for c in df.columns if 'NAME' in c or 'PLAYER' in c), None)
            status_col = next((c for c in df.columns if 'STATUS' in c), None)
            
            # Fallback for "Pos" column confusion
            if not status_col and len(df.columns) >= 3:
                # If Col 1 is Position, Col 2 is Status
                if str(df.iloc[0, 1]) in ['PG', 'SG', 'SF', 'PF', 'C']:
                    clean_df = df.iloc[:, [0, 2]].copy()
                    clean_df.columns = ['Player', 'Injury Status']
                    all_injuries.append(clean_df)
                    continue

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
        board = scoreboardv2.ScoreboardV2(game_date=date_str)
        header = board.game_header.get_data_frame()
        linescore = board.line_score.get_data_frame()
        if header.empty: return pd.DataFrame(), [], {}
        
        team_map = pd.Series(linescore.TEAM_ABBREVIATION.values, index=linescore.TEAM_ID).to_dict()
        
        opponent_map = {}
        games = []
        active_teams = []
        
        for _, row in header.iterrows():
            # CAST TO STRING IMMEDIATELY
            home_id = str(row['HOME_TEAM_ID'])
            away_id = str(row['VISITOR_TEAM_ID'])
            
            opponent_map[home_id] = away_id
            opponent_map[away_id] = home_id
            active_teams.extend([home_id, away_id])
            
            home_name = team_map.get(int(home_id), '???')
            away_name = team_map.get(int(away_id), '???')
            games.append({"Matchup": f"{away_name} @ {home_name}", "Status": row['GAME_STATUS_TEXT'].strip()})
            
        return pd.DataFrame(games), active_teams, opponent_map
    except:
        return pd.DataFrame(), [], {}

@st.cache_data
def get_roster_data(team_ids):
    all_rosters = []
    for tid in team_ids:
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=tid).get_data_frames()[0]
            # CAST TO STRING
            roster['PLAYER_ID'] = roster['PLAYER_ID'].astype(str)
            all_rosters.append(roster[['PLAYER_ID', 'POSITION']])
        except: continue
    
    if all_rosters: return pd.concat(all_rosters)
    return pd.DataFrame(columns=['PLAYER_ID', 'POSITION'])

@st.cache_data
def get_current_nba_season():
    """Calculates the correct 'YYYY-YY' string for the current NBA season."""
    now = datetime.now()
    # If we are in Jan-Sept (Month 1-9), the season started the previous year.
    # e.g., Jan 2026 is the '2025-26' season.
    if now.month < 10: 
        start_year = now.year - 1
    else:
        start_year = now.year
        
    end_year_short = str(start_year + 1)[-2:]
    return f"{start_year}-{end_year_short}"

def get_usage_and_defense(season_str=None): # Allow optional override
    """Fetches Advanced Stats for the correct season."""
    
    # 1. Auto-fix the season string if not valid or provided
    if not season_str:
        season_str = get_current_nba_season()
        
    print(f"🛠️ DEBUG: Fetching stats for season: '{season_str}'") # <--- CHECK THIS IN CONSOLE

    try:
        # headers = {'User-Agent': 'Mozilla/5.0'} # Uncomment if you still get timeout/403 errors
        
        # 2. Player Stats (Usage)
        # Added timeout to prevent hanging
        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str, 
            timeout=30
        ).get_data_frames()[0]
        
        usage_df = player_stats[['PLAYER_ID', 'USG_PCT']].copy()
        usage_df['PLAYER_ID'] = usage_df['PLAYER_ID'].astype(str)

        # 3. Team Stats (Defense)
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season_str, 
            timeout=30
        ).get_data_frames()[0]
        
        team_stats['Def_Rank'] = team_stats['DEF_RATING'].rank(ascending=True)
        defense_df = team_stats[['TEAM_ID', 'Def_Rank']].copy()
        defense_df['TEAM_ID'] = defense_df['TEAM_ID'].astype(str)

        print(f"✅ DEBUG: Success! Found {len(usage_df)} players.")
        return usage_df, defense_df

    except Exception as e:
        print(f"❌ STATS ERROR: {e}")
        # Use traceback to see exactly WHICH line failed
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

def calculate_dk_points(row):
    score = (row['PTS']) + (row['FG3M'] * 0.5) + (row['REB'] * 1.25) + (row['AST'] * 1.5) + (row['STL'] * 2) + (row['BLK'] * 2) - (row['TOV'] * 0.5)
    double_digits = sum(1 for s in [row['PTS'], row['REB'], row['AST'], row['STL'], row['BLK']] if s >= 10)
    if double_digits >= 3: score += 3
    elif double_digits >= 2: score += 1.5
    return score

@st.cache_data
def load_and_process_data(season_str):
    # Load requested season + previous season for history
    prev_season = f"{int(season_str[:4])-1}-{int(season_str[-2:])-1}"
    seasons = [prev_season, season_str]
    
    dfs = []
    for season in seasons:
        try:
            logs = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
            dfs.append(logs)
        except: pass
    
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # CAST IDS TO STRING
    df['PLAYER_ID'] = df['PLAYER_ID'].astype(str)
    df['TEAM_ID'] = df['TEAM_ID'].astype(str)
    
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
    
    # Calculate Correct Season based on Date
    current_season_str = get_season_string(selected_date)
    st.caption(f"Season: {current_season_str}")
    
    st.markdown("### 💰 Salary Data")
    salary_file = st.file_uploader("Upload DKSalaries.csv", type=['csv'])
    st.markdown("### 🛠️ Manual Controls")
    manual_remove_text = st.text_area("Paste OUT Players", height=100, placeholder="Joel Embiid\nKyrie Irving")
    
    schedule_df, active_teams, opponent_map = get_daily_schedule(selected_date)
    st.markdown("### 📅 Games Schedule")
    if not schedule_df.empty:
        for _, game in schedule_df.iterrows():
            st.markdown(f"<div class='game-card'><b>{game['Matchup']}</b><br><span style='color:#bbb'>{game['Status']}</span></div>", unsafe_allow_html=True)
    else:
        st.info("No games scheduled.")
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
        
        if not active_ids:
            st.error("⚠️ API Error: Could not fetch rosters.")
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
                    if manual_remove_text:
                        manual_names = [normalize_name(n) for n in manual_remove_text.split('\n') if n.strip()]
                        injured_list_norm.extend(manual_names)
                        for m_name in manual_names: status_map[m_name] = "OUT (Manual)"
                    st.metric("Injured Players", len(injured_list_norm))
            
            with status3: st.metric("Model Status", "Ready")
            
            with st.spinner(f"Fetching Advanced Stats ({current_season_str})..."):
                 usage_df, defense_df = get_usage_and_defense(current_season_str)

            with st.spinner("Running XGBoost Model..."):
                df = load_and_process_data(current_season_str)
                
                if not df.empty:
                    features = ['L5_DK_PTS', 'L10_DK_PTS', 'L5_MIN', 'L10_MIN', 'DAYS_REST', 'L5_FGA', 'L10_FGA']
                    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.1)
                    model.fit(df[features], df['DK_PTS'])
                    
                    latest = df.groupby('PLAYER_ID').tail(1).copy()
                    latest['PLAYER_NAME_NORM'] = latest['PLAYER_NAME'].apply(normalize_name)
                    mask_active = latest['PLAYER_ID'].isin(active_ids)
                    slate = latest[mask_active].copy()
                    
                    # --- MERGE DATA ---
                    # 1. Position
                    if not roster_df.empty:
                        slate = slate.merge(roster_df, on='PLAYER_ID', how='left')
                    
                    # 2. Usage (Fix: Force String Merge)
                    if not usage_df.empty:
                        slate = slate.merge(usage_df, on='PLAYER_ID', how='left')
                        # Fill NaNs with League Avg (20%) instead of 0
                        slate['USG_PCT'] = slate['USG_PCT'].fillna(0.20)
                    else: 
                        slate['USG_PCT'] = 0.20

                    # 3. Defense
                    slate['OPP_TEAM_ID'] = slate['TEAM_ID'].map(opponent_map)
                    if not defense_df.empty:
                        slate = slate.merge(defense_df, left_on='OPP_TEAM_ID', right_on='TEAM_ID', how='left')
                        slate['Def_Rank'] = slate['Def_Rank'].fillna(15)
                    else: 
                        slate['Def_Rank'] = 15

                    if not show_injured:
                        mask_auto_injury = ~slate['PLAYER_NAME_NORM'].isin(injured_list_norm)
                        slate = slate[mask_auto_injury]

                    if not slate.empty:
                        slate['Proj_DK_PTS'] = model.predict(slate[features])
                        slate['Injury Status'] = slate['PLAYER_NAME_NORM'].map(status_map).fillna("")

                        if salary_file is not None:
                            try:
                                salaries = pd.read_csv(salary_file)
                                if 'Salary' in salaries.columns:
                                    api_names = slate['PLAYER_NAME'].unique().tolist()
                                    salary_names = salaries['Name'].unique().tolist()
                                    name_mapping = smart_map_names(api_names, salary_names)
                                    slate['Matched_Name'] = slate['PLAYER_NAME'].map(name_mapping)
                                    slate = slate.merge(salaries[['Name', 'Salary']], left_on='Matched_Name', right_on='Name', how='left')
                                    
                                    slate['Salary'] = pd.to_numeric(slate['Salary'], errors='coerce').fillna(0)
                                    slate['Proj_DK_PTS'] = pd.to_numeric(slate['Proj_DK_PTS'], errors='coerce').fillna(0)
                                    slate = slate[slate['Salary'] > 0]
                                    
                                    bad_match_mask = (slate['Proj_DK_PTS'] > 30) & (slate['Salary'] < 4000)
                                    if bad_match_mask.any(): slate = slate[~bad_match_mask]
                                    
                                    slate['Value'] = slate.apply(lambda x: x['Proj_DK_PTS'] / (x['Salary']/1000), axis=1)
                                else: slate['Salary'] = 0; slate['Value'] = 0
                            except: slate['Salary'] = 0; slate['Value'] = 0
                        else: slate['Salary'] = 0; slate['Value'] = 0

                        slate = slate.sort_values(by='Proj_DK_PTS', ascending=False)
                        top_scorers = slate.head(3)
                        top_value = slate[slate['Proj_DK_PTS'] > 18].sort_values(by='Value', ascending=False).head(3)

                        def draw_card(player):
                            img = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player.PLAYER_ID}.png"
                            status_html = f"<div style='color:#FFC107; font-size:0.8em; margin-bottom:5px;'>⚠️ {player['Injury Status']}</div>" if player['Injury Status'] else ""
                            usg_val = player.get('USG_PCT', 0.2) * 100
                            pos = player.get('POSITION', 'UNK')
                            
                            card_html = (
                                f'<div style="text-align:center; background-color:#262730; padding:15px; border-radius:12px; border:1px solid #444;">'
                                f'<img src="{img}" style="width:90px; height:90px; border-radius:50%; border: 2px solid #333; margin-bottom:10px; object-fit: cover;" onerror="this.onerror=null; this.src=\'https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png\'">'
                                f'<div style="font-weight:bold; font-size:1.1em; margin-bottom:5px;">{player.PLAYER_NAME} <span style="font-size:0.8em; color:#bbb;">({pos})</span></div>'
                                f'{status_html}'
                                f'<div style="font-size:0.8em; color:#aaa; margin-bottom:5px;">Usage: {usg_val:.1f}%</div>'
                                f'<div style="display:flex; justify-content:space-between; background:#1e1e24; padding:8px 12px; border-radius:6px; margin-top:8px;">'
                                f'<span style="color:#ddd;">💰 ${int(player.Salary)}</span>'
                                f'<span style="color:#fff; font-weight:bold;">📊 {player.Proj_DK_PTS:.1f}</span>'
                                f'</div>'
                                f'<div style="color: #4CAF50; font-size:1.4em; font-weight:800; margin-top:10px;">'
                                f'{player.Value:.1f}x <span style="font-size:0.6em; font-weight:normal; color:#aaa;">VAL</span>'
                                f'</div></div>'
                            )
                            st.markdown(card_html, unsafe_allow_html=True)

                        st.markdown("### 🏆 Top 3 Projected Scorers")
                        c1, c2, c3 = st.columns(3)
                        for idx, col in enumerate([c1, c2, c3]):
                            if idx < len(top_scorers): 
                                with col: draw_card(top_scorers.iloc[idx])

                        st.markdown("---")
                        tab1, tab2, tab3 = st.tabs(["📋 Rankings", "📊 Teams", "🛠️ Debug"])
                        
                        with tab1:
                            search = st.text_input("🔍 Search", "")
                            cols = ['PLAYER_NAME', 'POSITION', 'TEAM_ABBREVIATION', 'Injury Status', 'Def_Rank', 'USG_PCT', 'Salary', 'Proj_DK_PTS', 'Value']
                            # Robust Column Selection
                            valid_cols = [c for c in cols if c in slate.columns]
                            show_df = slate[valid_cols].copy().reset_index(drop=True)
                            
                            if search: show_df = show_df[show_df['PLAYER_NAME'].str.contains(search, case=False)]
                            
                            def highlight_matchup(val):
                                if pd.isna(val): return ''
                                if val >= 20: return 'color: #4CAF50; font-weight: bold' 
                                if val <= 10: return 'color: #FF5252; font-weight: bold'
                                return ''

                            st.dataframe(show_df.style
                                         .format({'Salary': '${:.0f}', 'Proj_DK_PTS': '{:.1f}', 'Value': '{:.2f}x', 'USG_PCT': '{:.1%}', 'Def_Rank': '{:.0f}'})
                                         .map(highlight_matchup, subset=['Def_Rank'])
                                         .background_gradient(subset=['Value'], cmap='RdYlGn', vmin=3, vmax=6),
                                         use_container_width=True, height=800)
                        
                        with tab2: st.bar_chart(slate.groupby('TEAM_ABBREVIATION')['Proj_DK_PTS'].sum().sort_values(ascending=False))
                        
                        with tab3:
                            st.write("### Data Health Check")
                            if usage_df.empty: st.error("❌ Usage Data: Empty (API Failed)")
                            else: st.success(f"✅ Usage Data: {len(usage_df)} rows")
                            
                            if defense_df.empty: st.error("❌ Defense Data: Empty (API Failed)")
                            else: st.success(f"✅ Defense Data: {len(defense_df)} rows")
                            
                            st.write("### ID Type Check")
                            st.write(f"Slate ID Type: {slate['PLAYER_ID'].dtype}")
                            if not usage_df.empty: st.write(f"Usage ID Type: {usage_df['PLAYER_ID'].dtype}")

                else: st.error("Error: No historical data available.")

