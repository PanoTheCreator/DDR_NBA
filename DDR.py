import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leagueleaders

# -----------------------------
# Pondérations
# -----------------------------
W_STEAL = 1.5
W_BLOCK = 1.2
DELTA = 0.5
LEAGUE_AVG_DRTG = 112.0  # moyenne ligue (ajuster selon saison)

def safe_per36(value, minutes):
    return (value / minutes) * 36 if minutes and minutes > 0 else np.nan

# -----------------------------
# Récupération NBA API (steals, blocks, minutes)
# -----------------------------
@st.cache_data
def fetch_league_leaders(season="2024-25"):
    ll = leagueleaders.LeagueLeaders(season=season, season_type_all_star="Regular Season")
    df = ll.get_data_frames()[0]
    return df[['PLAYER','TEAM','GP','MIN','STL','BLK']].copy()

# -----------------------------
# Scraping Basketball Reference (DBPM + DRtg)
# -----------------------------
@st.cache_data
def fetch_bref_stats(season="2024"):
    url_adv = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
    url_misc = f"https://www.basketball-reference.com/leagues/NBA_{season}_misc.html"

    # DBPM
    try:
        df_adv = pd.read_html(url_adv)[0]
        if 'DBPM' in df_adv.columns:
            df_adv = df_adv[['Player','DBPM']].copy()
            df_adv.rename(columns={'Player':'PLAYER'}, inplace=True)
        else:
            df_adv = pd.DataFrame(columns=['PLAYER','DBPM'])
    except Exception:
        st.warning(f"Impossible de récupérer DBPM pour {season}.")
        df_adv = pd.DataFrame(columns=['PLAYER','DBPM'])

    # DRtg
    try:
        df_misc = pd.read_html(url_misc)[0]
        if 'DRtg' in df_misc.columns:
            df_misc = df_misc[['Player','DRtg']].copy()
            df_misc.rename(columns={'Player':'PLAYER'}, inplace=True)
        else:
            df_misc = pd.DataFrame(columns=['PLAYER','DRtg'])
    except Exception:
        st.warning(f"Impossible de récupérer DRtg pour {season}.")
        df_misc = pd.DataFrame(columns=['PLAYER','DRtg'])

    # Fusion
    df_full = pd.merge(df_adv, df_misc, on='PLAYER', how='outer')
    return df_full

# -----------------------------
# Calcul DDR
# -----------------------------
def compute_ddr(df_indiv, df_bref):
    df = pd.merge(df_indiv, df_bref, on='PLAYER', how='left')

    df['STL'] = df['STL'].fillna(0)
    df['BLK'] = df['BLK'].fillna(0)

    df['RAW_DDR_EVENTS'] = W_STEAL*df['STL'] + W_BLOCK*df['BLK']
    df['DDR_per36'] = df.apply(lambda r: safe_per36(r['RAW_DDR_EVENTS'], r['MIN']), axis=1)

    df['DRtg_eff'] = df['DRtg'].fillna(LEAGUE_AVG_DRTG)
    df['DRtg_factor'] = (LEAGUE_AVG_DRTG / df['DRtg_eff']).clip(lower=0.7, upper=1.3)

    df['DBPM_eff'] = df['DBPM'].fillna(0.0)

    df['DDR_final'] = df['DDR_per36'] * df['DRtg_factor'] + (DELTA * df['DBPM_eff'])

    return df[['PLAYER','TEAM','MIN','STL','BLK','DRtg','DBPM','DDR_per36','DDR_final']].sort_values('DDR_final', ascending=False)

# -----------------------------
# 🎨 Interface Streamlit avec style violet
# -----------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #6A0DAD;
    }
    .main {
        background-color: #6A0DAD;
    }
    h1 {
        text-align: center;
        color: white;
        font-size: 50px;
    }
    .stButton button {
        display: block;
        margin: 0 auto;
        background-color: white;
        color: #6A0DAD;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 🏀 Page d'accueil
st.markdown("<h1>DDR powered by Pano</h1>", unsafe_allow_html=True)

if st.button("Accéder à l'application"):
    st.success("🚀 Lancement de la génération DDR !")

    season = st.text_input("Saison NBA API (ex: 2024-25)", value="2024-25")
    season_bref = st.text_input("Saison Basketball Reference (ex: 2024)", value="2024")

    # Chargement des données
    df_indiv = fetch_league_leaders(season)
    df_bref = fetch_bref_stats(season_bref)
    df_ddr = compute_ddr(df_indiv, df_bref)

    # --- Filtres interactifs ---
    teams = sorted(df_ddr['TEAM'].dropna().unique())
    selected_team = st.selectbox("Choisir une équipe", options=["Tous les joueurs"] + list(teams))
    min_threshold = st.slider("Filtrer par minutes jouées (minimum)", 500, int(df_ddr['MIN'].max()), 2000)

    # Application des filtres
    df_filtered = df_ddr.copy()
    if selected_team != "Tous les joueurs":
        df_filtered = df_filtered[df_filtered['TEAM'] == selected_team]
    df_filtered = df_filtered[df_filtered['MIN'] >= min_threshold]

    # Affichage
    st.subheader("Classement DDR filtré")
    st.dataframe(df_filtered.head(20))

    st.subheader("DDR_final vs Minutes")
    st.scatter_chart(df_filtered.set_index("PLAYER")[["MIN","DDR_final"]])
