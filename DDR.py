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
W_PF = -1.0  # les fautes pénalisent le DDR

def safe_per36(value, minutes):
    return (value / minutes) * 36 if minutes and minutes > 0 else np.nan

# -----------------------------
# Récupération NBA API (steals, blocks, minutes, fautes)
# -----------------------------
@st.cache_data
def fetch_league_leaders(season="2024-25"):
    ll = leagueleaders.LeagueLeaders(season=season, season_type_all_star="Regular Season")
    df = ll.get_data_frames()[0]
    return df[['PLAYER','TEAM','GP','MIN','STL','BLK','PF']].copy()

# -----------------------------
# Chargement du fichier opp_pts_poss.xlsx
# -----------------------------
@st.cache_data
def fetch_opp_pts_poss():
    try:
        df = pd.read_excel("opp_pts_poss.xlsx")
        df.columns = df.columns.str.strip().str.lower()  # standardise les noms
        df = df.rename(columns={"player": "PLAYER", "oppptsposs": "opp_pts_poss"})
        st.success("✅ Fichier opp_pts_poss.xlsx chargé avec succès.")
        st.write("Colonnes détectées :", list(df.columns))
        st.write("Aperçu des 5 premières lignes :")
        st.dataframe(df.head())
        return df[['PLAYER','opp_pts_poss']]
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier : {e}")
        return pd.DataFrame(columns=['PLAYER','opp_pts_poss'])

# -----------------------------
# Calcul DDR
# -----------------------------
def compute_ddr(df_indiv, df_opp):
    df = pd.merge(df_indiv, df_opp, on='PLAYER', how='left')

    df['STL'] = df['STL'].fillna(0)
    df['BLK'] = df['BLK'].fillna(0)
    df['PF'] = df['PF'].fillna(0)
    df['opp_pts_poss'] = df['opp_pts_poss'].fillna(1.0)  # éviter division par zéro

    # Événements défensifs pondérés
    df['RAW_DDR_EVENTS'] = W_STEAL*df['STL'] + W_BLOCK*df['BLK'] + W_PF*df['PF']

    # Normalisation par 36 minutes
    df['DDR_per36'] = df.apply(lambda r: safe_per36(r['RAW_DDR_EVENTS'], r['MIN']), axis=1)

    # Ajustement par points concédés par possession
    df['DDR_final'] = df['DDR_per36'] / df['opp_pts_poss']

    return df[['PLAYER','TEAM','MIN','STL','BLK','PF','opp_pts_poss','DDR_per36','DDR_final']].sort_values('DDR_final', ascending=False)

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
    df_opp = fetch_opp_pts_poss()
    df_ddr = compute_ddr(df_indiv, df_opp)

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
        tooltip=['PLAYER','TEAM','MIN','DDR_final','PF','opp_pts_poss']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
