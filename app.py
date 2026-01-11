import streamlit as st
import pandas as pd
import xgboost as xgb
import requests
import unicodedata
import difflib
import io
import textwrap
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
    div[data-testid="stExpander"] details summary p { font-size: 1.1em; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA FUNCTIONS ---

def normalize_name(name):
    """Standardizes names for matching."""
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
    """Maps API names to Salary CSV names."""
    mapping = {}
    salary_norm_map = {normalize_name(name): name for name in salary_names}
    salary_norm_list = list(salary_norm_map.keys())
    
    for api_name in api_names:
        norm_api = normalize_name(api_name)
        if norm_api in salary_norm_map:
            mapping[api_name] = salary_norm_map[norm_api]
        else:
            # Force first letter match + 90% similarity
            candidates = [n for n in salary_norm_list if n.startswith(norm_api[0])]
            matches = difflib.get_close_matches(norm_api, candidates, n=1, cutoff=0.90)
            if matches:
                mapping[api_name] = salary_norm_map[matches[0]]
    return mapping

@st.cache_data
def get_injury_report():
    """
    STRICT ESPN SCRAPER
    - Grabs ALL tables from espn.com/nba/injuries
    - forces Col 0 -> Player, Col 1 -> Status
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    url = "https://www.espn.com/nba/injuries"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        # ESPN has 30 separate tables (one per team)
        dfs = pd.read_html(io.StringIO(response.text))
        
        all_injuries = []
        for df in dfs:
            # Basic validation: Table must have at least 2 columns (Name, Status)
            if len(df.columns) >= 2:
                # Force rename the first two columns regardless of what ESPN calls them
                # This fixes issues where 'NAME' header is missing
                clean_df = df.iloc[:, :2].copy()
                clean_df.columns = ['Player', 'Injury Status']
                
                # Filter out header rows that got scraped as data
                clean_df = clean_df[clean_df['Player'] != 'NAME']
                clean_df = clean_df[clean_df['Player'] != 'Player']
                
                all_injuries.append(clean_df)
        
        if all_injuries:
            combined = pd.concat(all_injuries)
            combined['Player_Norm'] = combined['Player'].apply(normalize_name)
            # Standardize status text
            return combined[['Player_Norm', 'Injury Status']]
            
    except Exception as e:
        print(f"ESPN Error: {e}")

    return pd.DataFrame(columns=['Player_Norm', 'Injury Status'])

@st.cache_data
def get_daily_schedule(selected_date):
    date_str = selected_date.strftime('%Y-%m-%d')
    try:
        board = scoreboardv2.ScoreboardV2(game_date=date_str)
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

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/0/03/National_Basketball_Association_logo.svg", width=80)
    st.title("CourtVision DFS")
    est = pytz.timezone('US/Eastern')
    selected_date = st.date_input("📅 Game Date", datetime.now(est))
    st.markdown("### 💰 Salary Data")
    salary_file = st.file_uploader("Upload DKSalaries.csv", type=['csv'])
    st.markdown("### 🛠️ Manual Controls")
    manual_remove_text = st.text_area("Paste OUT Players (One per line)", height=100, placeholder="Joel Embiid\nKyrie Irving")
    
    schedule_df, active_teams = get_daily_schedule(selected_date)
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
        
        active_ids = []
        with st.spinner("Fetching Rosters..."): active_ids = get_roster_players(active_teams)
        
        if not active_ids:
            st.error("⚠️ API Error: Could not fetch rosters.")
        else:
            with status2:
                with st.spinner("Checking Injuries (ESPN)..."):
                    injury_df = get_injury_report()
                    
                    if not injury_df.empty:
                        # Define what counts as "Injured" for exclusion
                        # For display, we keep everything. For filtering, we look for Out/Doubtful.
                        status_map = dict(zip(injury_df['Player_Norm'], injury_df['Injury Status']))
                        
                        # Calculate count of strictly OUT players
                        exclude_mask = injury_df['Injury Status'].str.contains('Out|Doubtful|Injured Reserve', case=False, na=False)
                        injured_list_norm = injury_df[exclude_mask]['Player_Norm'].tolist()
                    else:
                        injured_list_norm = []
                        status_map = {}
                    
                    if manual_remove_text:
                        manual_names = [normalize_name(n) for n in manual_remove_text.split('\n') if n.strip()]
                        injured_list_norm.extend(manual_names)
                        for m_name in manual_names: status_map[m_name] = "OUT (Manual)"

                    st.metric("Injured Players", len(injured_list_norm))
            
            with status3: st.metric("Model Status", "Ready")
            
            with st.spinner("Running XGBoost Model..."):
                df = load_and_process_data()
                
                if not df.empty:
                    features = ['L5_DK_PTS', 'L10_DK_PTS', 'L5_MIN', 'L10_MIN', 'DAYS_REST', 'L5_FGA', 'L10_FGA']
                    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.1)
                    model.fit(df[features], df['DK_PTS'])
                    
                    latest = df.groupby('PLAYER_ID').tail(1).copy()
                    latest['PLAYER_NAME_NORM'] = latest['PLAYER_NAME'].apply(normalize_name)
                    mask_active = latest['PLAYER_ID'].isin(active_ids)
                    slate = latest[mask_active].copy()

                    if not show_injured:
                        mask_auto_injury = ~slate['PLAYER_NAME_NORM'].isin(injured_list_norm)
                        slate = slate[mask_auto_injury]

                    if not slate.empty:
                        slate['Proj_DK_PTS'] = model.predict(slate[features])
                        # Map injury status to the slate
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
                                    if bad_match_mask.any():
                                        st.toast(f"⚠️ Removed {bad_match_mask.sum()} suspicious matches", icon="🧹")
                                        slate = slate[~bad_match_mask]
                                    
                                    slate['Value'] = slate.apply(lambda x: x['Proj_DK_PTS'] / (x['Salary']/1000), axis=1)
                                else:
                                    slate['Salary'] = 0; slate['Value'] = 0
                            except: slate['Salary'] = 0; slate['Value'] = 0
                        else:
                            slate['Salary'] = 0; slate['Value'] = 0

                        slate = slate.sort_values(by='Proj_DK_PTS', ascending=False)
                        top_scorers = slate.head(3)
                        top_value = slate[slate['Proj_DK_PTS'] > 18].sort_values(by='Value', ascending=False).head(3)
                        bad_plays = slate[slate['Salary'] > 6000].sort_values(by='Value', ascending=True).head(3)

                        def draw_card(player, label_type="points"):
                            img = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player.PLAYER_ID}.png"
                            status_html = f"<div style='color:#FFC107; font-size:0.8em; margin-bottom:5px;'>⚠️ {player['Injury Status']}</div>" if player['Injury Status'] else ""
                            color = '#4CAF50' if label_type!='bad' else '#FF5252'
                            
                            card_html = textwrap.dedent(f"""
                                <div style="text-align:center; background-color:#262730; padding:15px; border-radius:12px; border:1px solid #444; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                                    <img src="{img}" style="width:90px; height:90px; border-radius:50%; border: 2px solid #333; margin-bottom:10px; object-fit: cover;" onerror="this.onerror=null; this.src='https://cdn.nba.com/headshots/nba/latest/1040x760/fallback.png'">
                                    <div style="font-weight:bold; font-size:1.1em; margin-bottom:5px;">{player.PLAYER_NAME}</div>
                                    {status_html}
                                    <div style="display:flex; justify-content:space-between; background:#1e1e24; padding:8px 12px; border-radius:6px; margin-top:8px;">
                                        <span style="color:#ddd;">💰 ${int(player.Salary)}</span>
                                        <span style="color:#fff; font-weight:bold;">📊 {player.Proj_DK_PTS:.1f}</span>
                                    </div>
                                    <div style="color: {color}; font-size:1.4em; font-weight:800; margin-top:10px;">
                                        {player.Value:.1f}x <span style="font-size:0.6em; font-weight:normal; color:#aaa;">VAL</span>
                                    </div>
                                </div>
                            """)
                            st.markdown(card_html, unsafe_allow_html=True)

                        st.markdown("### 🏆 Top 3 Projected Scorers")
                        c1, c2, c3 = st.columns(3)
                        for idx, col in enumerate([c1, c2, c3]):
                            if idx < len(top_scorers): 
                                with col: draw_card(top_scorers.iloc[idx])

                        if salary_file:
                            st.markdown("---")
                            col_v, col_b = st.columns(2)
                            with col_v:
                                st.markdown("### 💎 Top Value (Proj > 18)")
                                cols = st.columns(3)
                                for i in range(min(3, len(top_value))):
                                    with cols[i]: draw_card(top_value.iloc[i], "value")
                            with col_b:
                                st.markdown("### 🛑 Top Fades (Salary > $6k)")
                                cols = st.columns(3)
                                for i in range(min(3, len(bad_plays))):
                                    with cols[i]: draw_card(bad_plays.iloc[i], "bad")

                        st.markdown("---")
                        tab1, tab2, tab3 = st.tabs(["📋 Rankings", "📊 Teams", "🛠️ Debug"])
                        
                        with tab1:
                            search = st.text_input("🔍 Search", "")
                            cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'Injury Status', 'Salary', 'Proj_DK_PTS', 'Value', 'L5_DK_PTS']
                            show_df = slate[cols].copy().reset_index(drop=True)
                            if search: show_df = show_df[show_df['PLAYER_NAME'].str.contains(search, case=False)]
                            
                            def color_injury(val):
                                if 'Out' in str(val) or 'Doubtful' in str(val): return 'color: #FF5252; font-weight: bold'
                                elif 'Questionable' in str(val) or 'Day' in str(val): return 'color: #FFC107; font-weight: bold'
                                return ''

                            st.dataframe(show_df.style.format({'Salary': '${:.0f}', 'Proj_DK_PTS': '{:.1f}', 'Value': '{:.2f}x', 'L5_DK_PTS': '{:.1f}'})
                                         .map(color_injury, subset=['Injury Status'])
                                         .background_gradient(subset=['Value'], cmap='RdYlGn', vmin=3, vmax=6),
                                         use_container_width=True, height=800)
                        
                        with tab2: st.bar_chart(slate.groupby('TEAM_ABBREVIATION')['Proj_DK_PTS'].sum().sort_values(ascending=False))
                        with tab3:
                            if salary_file:
                                debug_df = slate[['PLAYER_NAME', 'Matched_Name', 'Salary', 'Proj_DK_PTS']]
                                st.dataframe(debug_df[debug_df['PLAYER_NAME'] != debug_df['Matched_Name']])
                else: st.error("Error: No historical data available.")
