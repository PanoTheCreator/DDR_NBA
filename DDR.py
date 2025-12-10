import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leagueleaders
import altair as alt

# -----------------------------
# Pondérations
# -----------------------------
W_STEAL = 1.5
W_BLOCK = 1.2

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
# Calcul DDR (simplifié sans DBPM/DRtg)
# -----------------------------
def compute_ddr(df_indiv):
    df = df_indiv.copy()
    df['STL'] = df['STL'].fillna(0)
    df['BLK'] = df['BLK'].fillna(0)

    df['RAW_DDR_EVENTS'] = W_STEAL*df['STL'] + W_BLOCK*df['BLK']
    df['DDR_per36'] = df.apply(lambda r: safe_per36(r['RAW_DDR_EVENTS'], r['MIN']), axis=1)

    # Score final = DDR_per36 (simple version)
    df['DDR_final'] = df['DDR_per36']

    return df[['PLAYER','TEAM','MIN','STL','BLK','DDR_per36','DDR_final']].sort_values('DDR_final', ascending=False)

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

    # Chargement des données
    df_indiv = fetch_league_leaders(season)
    df_ddr = compute_ddr(df_indiv)

    # --- Filtres interactifs ---
    teams = sorted(df_ddr['TEAM'].dropna().unique())
    selected_team = st.selectbox("Choisir une équipe", options=["Tous les joueurs"] + list(teams))
    min_threshold = st.slider("Filtrer par minutes jouées (minimum)", 0, int(df_ddr['MIN'].max()), 15)

    # Application des filtres
    df_filtered = df_ddr.copy()
    if selected_team != "Tous les joueurs":
        df_filtered = df_filtered[df_filtered['TEAM'] == selected_team]
    df_filtered = df_filtered[df_filtered['MIN'] >= min_threshold]

    # Affichage tableau
    st.subheader("Classement DDR filtré")
    st.dataframe(df_filtered)

    # Scatter plot Altair
    st.subheader("Scatter Plot : DDR_final vs Minutes")
    chart = alt.Chart(df_filtered).mark_circle(size=80).encode(
        x=alt.X('MIN', title='Minutes'),
        y=alt.Y('DDR_final', title='DDR Final'),
        color=alt.Color('TEAM', title='Équipe'),
        tooltip=['PLAYER','TEAM','MIN','DDR_final']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
