import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leagueleaders
import altair as alt

# -----------------------------
# Pondérations
# -----------------------------
W_STEAL = 1.8
W_BLOCK = 1.4
W_FOUL = -1.5

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
# Chargement OppPtsPoss depuis Excel
# -----------------------------
@st.cache_data
def fetch_opp_excel(path="opp_pts_poss.xlsx"):
    df_opp = pd.read_excel(path)

    # Nettoyage des noms de colonnes : supprime espaces et met en majuscules
    df_opp.columns = df_opp.columns.str.strip().str.upper()

    # Vérification : si la colonne PLAYER existe bien
    if 'PLAYER' not in df_opp.columns:
        st.error("Le fichier Excel doit contenir une colonne 'PLAYER'. Colonnes trouvées : " + str(df_opp.columns.tolist()))
        return pd.DataFrame(columns=['PLAYER','OPPPTSPOSS'])

    return df_opp

# -----------------------------
# Calcul DDR
# -----------------------------
def compute_ddr(df_indiv, df_opp):
    df = pd.merge(df_indiv, df_opp, on='PLAYER', how='left')

    # Remplissage valeurs manquantes
    df[['STL','BLK','PF']] = df[['STL','BLK','PF']].fillna(0)
    df['OPPPTSPOSS'] = df['OPPPTSPOSS'].fillna(0.0)

    # Événements perturbateurs avec fautes négatives
    df['RAW_DDR_EVENTS'] = (
        W_STEAL*df['STL'] +
        W_BLOCK*df['BLK'] +
        W_FOUL*df['PF']
    )

    # DDR par 36 minutes
    df['DDR_per36'] = df.apply(lambda r: safe_per36(r['RAW_DDR_EVENTS'], r['MIN']), axis=1)

    # DDR par possessions
    df['DDR_per75'] = df['DDR_per36'] * (75/36)
    df['DDR_per100'] = df['DDR_per36'] * (100/36)

    # Facteur OppPtsPoss (impact défensif direct)
    df['OppFactor'] = 1.3 - (df['OPPPTSPOSS'] / 100.0)

    # DDR final
    df['DDR_final'] = df['DDR_per36'] * df['OppFactor']

    # Split nom/prénom robuste
    df['Prénom'] = df['PLAYER'].str.split().str[0]
    df['Nom'] = df['PLAYER'].str.split().str[1:].str.join(' ')

    # Colonnes finales
    df_final = df[['Prénom','Nom','TEAM','MIN','DDR_per36','DDR_per75','DDR_per100','DDR_final']]

    return df_final.sort_values('DDR_final', ascending=False)

# -----------------------------
# Interface Streamlit
# -----------------------------
st.title("Defensive Disruption Rate (DDR) – Version OppPtsPoss")

season = st.text_input("Saison NBA API (ex: 2024-25)", value="2024-25")
min_threshold = st.slider("Minutes minimum", 0, 500, 1000, 2000)
selected_team = st.text_input("Équipe (laisser vide pour toutes)", value="")

if st.button("Générer DDR"):
    with st.spinner("Chargement des données..."):
        df_indiv = fetch_league_leaders(season)
        df_opp = fetch_opp_excel("opp_pts_poss.xlsx")
        df_ddr = compute_ddr(df_indiv, df_opp)

        # Filtres
        if selected_team.strip():
            df_ddr = df_ddr[df_ddr['TEAM'] == selected_team]
        df_ddr = df_ddr[df_ddr['MIN'] >= min_threshold]

        # Afficher directement tout le classement
        st.subheader("Classement DDR complet")
        st.dataframe(df_ddr)

        # Bouton de téléchargement CSV
        st.download_button("Télécharger le classement complet",
                           df_ddr.to_csv(index=False).encode('utf-8'),
                           "DDR_classement.csv",
                           "text/csv")

        # Scatter plot DDR_final vs DDR_per100 avec labels interactifs
        st.subheader("Scatter Plot : DDR_final vs DDR/100 poss")

        chart = alt.Chart(df_ddr).mark_circle(size=80).encode(
            x=alt.X('DDR_final', title='DDR Final'),
            y=alt.Y('DDR_per100', title='DDR/100 poss'),
            color=alt.Color('TEAM', title='Équipe'),  # coloration par équipe
            tooltip=['Prénom','Nom','TEAM','MIN','DDR_final','DDR_per100']
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
