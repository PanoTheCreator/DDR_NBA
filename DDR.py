import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leagueleaders
import altair as alt

# -----------------------------
# Pondérations ajustées
# -----------------------------
W_STEAL = 1.0
W_BLOCK = 0.8
W_FOUL = -1.2
W_DEFLECTION = 0.5

# -----------------------------
# Chargement OppPtsPoss + % + deflections depuis Excel
# -----------------------------
def fetch_opp_excel(path):
    df_opp = pd.read_excel(path)
    df_opp.columns = df_opp.columns.str.strip().str.upper()

    df_opp = df_opp.rename(columns={
        'OPP_PTS_POSS': 'OPPPTSPOSS',
        'DEFLECTIONS': 'DEFLECTIONS',
        'FOUL%': 'PF%',
        'STL%': 'STL%',
        'BLK%': 'BLK%',
        'OPP_EFG%': 'OPP_EFG%',
        'OPP_TOV%': 'OPP_TOV%',
        'OPP_ORB%': 'OPP_ORB%',
        'OPP_FT RATE': 'OPP_FTR'
    })

    # Conversion en numérique
    for col in ['STL%','BLK%','PF%','DEFLECTIONS','OPPPTSPOSS','OPP_EFG%','OPP_TOV%','OPP_ORB%','OPP_FTR']:
        if col in df_opp.columns:
            df_opp[col] = (
                df_opp[col]
                .astype(str)
                .str.replace('%','')
                .str.replace(',','.')
            )
            df_opp[col] = pd.to_numeric(df_opp[col], errors='coerce')

    # Convertir % en décimales
    for col in ['STL%','BLK%','PF%','OPP_EFG%','OPP_TOV%','OPP_ORB%','OPP_FTR']:
        if col in df_opp.columns:
            df_opp[col] = df_opp[col] / 70.0

    # Harmonisation des noms
    df_opp['PLAYER'] = df_opp['PLAYER'].str.strip().str.upper()
    df_opp = df_opp.drop_duplicates(subset='PLAYER', keep='first')

    return df_opp

# -----------------------------
# Calcul DDR unifié
# -----------------------------
def compute_ddr(df_indiv, df_opp):
    df_indiv['PLAYER'] = df_indiv['PLAYER'].str.strip().str.upper()
    df = pd.merge(df_indiv, df_opp, on='PLAYER', how='left')

    for col in ['STL','BLK','PF','MIN','GP','OPPPTSPOSS','STL%','BLK%','PF%','DEFLECTIONS','OPP_EFG%','OPP_TOV%','OPP_ORB%','OPP_FTR']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # DDR-E (efficacité pondérée, normalisée)
    df['DDR-E'] = (
        W_STEAL * df['STL%'] +
        W_BLOCK * df['BLK%'] +
        W_FOUL  * df['PF%']
    ) * 120  # facteur d’échelle pour atteindre ~ -3 à +10

    # Volumes
    df['VolPos'] = W_STEAL * df['STL'] + W_BLOCK * df['BLK'] + W_DEFLECTION * df['DEFLECTIONS']
    df['VolNeg'] = abs(W_FOUL) * df['PF']

    # Contexte individuel
    df['ContextE'] = np.where(df['DDR-E'] > 0, 1.1, 0.9)

    # Contexte collectif enrichi
    df['ContextTeam'] = (
        (1 - df['OPP_EFG%']) * 1.1 +
        df['OPP_TOV%'] * 1.3 +
        (1 - df['OPP_ORB%']) * 1.1 +
        (1 - df['OPP_FTR']) * 1.2
    )

    # DDR final (compressé et normalisé)
    df['DDR'] = np.where(
        df['VolNeg'] != 0,
        (np.sqrt(df['VolPos'] / df['VolNeg']) * df['ContextE'] * df['ContextTeam']) * 2,
        np.nan
    )

    df['Prénom'] = df['PLAYER'].str.split().str[0].str.capitalize()
    df['Nom'] = df['PLAYER'].str.split().str[1:].str.join(' ').str.capitalize()

    df['Rank DDR-E'] = df['DDR-E'].rank(ascending=False, method='min').fillna(0).astype(int)
    df['Rank DDR'] = df['DDR'].rank(ascending=False, method='min').fillna(0).astype(int)

    return df[['Prénom','Nom','TEAM','MIN','DDR-E','Rank DDR-E','DDR','Rank DDR']].sort_values('DDR', ascending=False)

# -----------------------------
# Interface Streamlit
# -----------------------------
st.title("Defensive Disruption Rate (DDR) -- Saison sélectionnable")

st.info("""
🧾 **DDR enrichi avec les 4 facteurs défensifs**

- **DDR‑E (Efficiency)** : efficacité individuelle pondérée par possession.  
- **DDR (Final)** : rapport VolPos/VolNeg corrigé par double contexte (individuel + collectif).  

Échelle cible : environ -3 à +10 pour les deux scores.
""")

# Menu déroulant pour choisir la saison
season = st.selectbox(
    "Choisir la saison NBA",
    options=["2024-25", "2025-26"],
    index=1
)

min_threshold = st.slider("Minutes minimum", 0, 2000, 500, 50)
selected_team = st.text_input("Équipe (laisser vide pour toutes)", value="")

@st.cache_data
def fetch_league_leaders(season="2025-26"):
    ll = leagueleaders.LeagueLeaders(season=season, season_type_all_star="Regular Season")
    df = ll.get_data_frames()[0]
    return df[['PLAYER','TEAM','GP','MIN','STL','BLK','PF']].copy()

if st.button("Générer DDR"):
    with st.spinner("Chargement des données..."):
        df_indiv = fetch_league_leaders(season)

        if season == "2025-26":
            df_opp = fetch_opp_excel("opp_pts_poss25-26.xlsx")
        else:
            df_opp = fetch_opp_excel("opp_pts_poss24_25.xlsx")

        df_ddr = compute_ddr(df_indiv, df_opp)

        if 'TEAM' in df_ddr.columns and selected_team.strip():
            df_ddr = df_ddr[df_ddr['TEAM'] == selected_team]
        df_ddr = df_ddr[df_ddr['MIN'] >= min_threshold]

        st.subheader(f"Classement DDR enrichi ({season})")
        st.dataframe(df_ddr)

        st.download_button(
            f"Télécharger le classement complet ({season})",
            df_ddr.to_csv(index=False).encode('utf-8'),
            f"DDR_{season}.csv",
            "text/csv"
        )

        # Scatter plot
        st.subheader("Scatter : DDR vs DDR-E")
        chart = alt.Chart(df_ddr).mark_circle(size=80).encode(
            x=alt.X('DDR', title='DDR (VolPos/VolNeg × ContextE × ContextTeam)'),
            y=alt.Y('DDR-E', title='DDR-E (efficacité pondérée)'),
            color=alt.Color('Nom', title='Joueur'),
            tooltip=['Prénom','Nom','TEAM','MIN','DDR','Rank DDR','DDR-E','Rank DDR-E']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

        # Histogramme
        st.subheader("Histogramme de la distribution des DDR")
        hist = alt.Chart(df_ddr).mark_bar().encode(
            alt.X("DDR", bin=alt.Bin(maxbins=30), title="DDR"),
            alt.Y("count()", title="Nombre de joueurs"),
            tooltip=["count()"]
        ).properties(width=600, height=400)
        st.altair_chart(hist, use_container_width=True)
