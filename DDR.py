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
W_DEFLECTION = 1.0  # ajusté pour éviter que ça domine trop

# -----------------------------
# Chargement OppPtsPoss + % + deflections depuis Excel
# -----------------------------
def fetch_opp_excel(path="opp_pts_poss24_25.xlsx"):
    df_opp = pd.read_excel(path)

    # Normalise tout en majuscules
    df_opp.columns = df_opp.columns.str.strip().str.upper()

    # Harmonisation des noms
    df_opp = df_opp.rename(columns={
        'OPP_PTS_POSS': 'OPPPTSPOSS',
        'DEFLECTIONS': 'DEFLECTIONS',
        'FOUL%': 'PF%',
        'STL%': 'STL%',
        'BLK%': 'BLK%'
    })

    # Conversion en numérique
    for col in ['STL%','BLK%','PF%','DEFLECTIONS','OPPPTSPOSS']:
        if col in df_opp.columns:
            df_opp[col] = (
                df_opp[col]
                .astype(str)
                .str.replace('%','')
                .str.replace(',','.')
            )
            df_opp[col] = pd.to_numeric(df_opp[col], errors='coerce')

    # Convertir % en décimales
    for col in ['STL%','BLK%','PF%']:
        if col in df_opp.columns:
            df_opp[col] = df_opp[col] / 100.0

    # Vérifie colonnes nécessaires
    required = ['PLAYER','OPPPTSPOSS','STL%','BLK%','PF%','DEFLECTIONS']
    missing = [c for c in required if c not in df_opp.columns]
    if missing:
        st.error(f"Colonnes manquantes dans Excel: {missing}. Colonnes trouvées: {df_opp.columns.tolist()}")
        for c in missing:
            df_opp[c] = 0.0
        if 'PLAYER' not in df_opp.columns:
            df_opp['PLAYER'] = ""
    return df_opp

# -----------------------------
# Calcul DDR unifié
# -----------------------------
def compute_ddr(df_indiv, df_opp):
    df = pd.merge(df_indiv, df_opp, on='PLAYER', how='left')

    # Remplissage valeurs manquantes
    for col in ['STL','BLK','PF','MIN','GP','OPPPTSPOSS','STL%','BLK%','PF%','DEFLECTIONS']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # DDR% (efficacité relative)
    df['DDR%'] = (
        W_STEAL * df['STL%'] +
        W_BLOCK * df['BLK%'] +
        W_FOUL  * df['PF%']
    )

    # DDR-V (volume par match)
    df['DDR-V'] = (
        W_STEAL * (df['STL'] / df['GP']) +
        W_BLOCK * (df['BLK'] / df['GP']) +
        W_FOUL  * (df['PF']  / df['GP']) +
        W_DEFLECTION * (df['DEFLECTIONS'] / df['GP'])
    )

    # Facteur contexte borné entre 0.6 et 1.4
    df['OppFactor'] = 1.3 - (df['OPPPTSPOSS'] / 100.0)
    df['OppFactor'] = df['OppFactor'].clip(lower=0.6, upper=1.4)

    # DDR final = moyenne des deux composantes * contexte
    df['DDR'] = ((df['DDR%'] + df['DDR-V']) / 2) * df['OppFactor']

    # Split nom/prénom
    df['Prénom'] = df['PLAYER'].str.split().str[0]
    df['Nom'] = df['PLAYER'].str.split().str[1:].str.join(' ')

    # Colonnes finales réduites
    df_final = df[['Prénom','Nom','MIN','DDR%','DDR-V','DDR']]
    return df_final.sort_values('DDR', ascending=False)

# -----------------------------
# Interface Streamlit
# -----------------------------
st.title("Defensive Disruption Rate (DDR) by Pano — Unifié")

season = st.text_input("Saison NBA API (ex: 2024-25)", value="2024-25")
min_threshold = st.slider("Minutes minimum", 0, 2000, 500, 50)
selected_team = st.text_input("Équipe (laisser vide pour toutes)", value="")

@st.cache_data
def fetch_league_leaders(season="2024-25"):
    ll = leagueleaders.LeagueLeaders(season=season, season_type_all_star="Regular Season")
    df = ll.get_data_frames()[0]
    return df[['PLAYER','TEAM','GP','MIN','STL','BLK','PF']].copy()

if st.button("Générer DDR"):
    with st.spinner("Chargement des données..."):
        df_indiv = fetch_league_leaders(season)
        df_opp = fetch_opp_excel("opp_pts_poss24_25.xlsx")  # ton fichier Excel

        # Calcul DDR
        df_ddr = compute_ddr(df_indiv, df_opp)

        # Filtres
        if selected_team.strip():
            df_ddr = df_ddr[df_ddr['TEAM'] == selected_team]
        df_ddr = df_ddr[df_ddr['MIN'] >= min_threshold]

        st.subheader("Classement DDR unifié")
        st.dataframe(
            df_ddr,
            column_config={
                "DDR": st.column_config.NumberColumn(
                    "DDR",
                    help="Score final (moyenne efficacité + volume, corrigée par contexte)",
                    format="%.2f",
                    min_value=df_ddr["DDR"].min(),
                    max_value=df_ddr["DDR"].max()
                ),
                "DDR%": st.column_config.NumberColumn(
                    "DDR%",
                    help="Efficacité défensive (pourcentages en décimales, fautes négatives)",
                    format="%.3f",
                    min_value=df_ddr["DDR%"].min(),
                    max_value=df_ddr["DDR%"].max()
                ),
                "DDR-V": st.column_config.NumberColumn(
                    "DDR-V",
                    help="Volume défensif par match (fautes négatives)",
                    format="%.2f",
                    min_value=df_ddr["DDR-V"].min(),
                    max_value=df_ddr["DDR-V"].max()
                )
            }
        )

        st.download_button(
            "Télécharger le classement complet",
            df_ddr.to_csv(index=False).encode('utf-8'),
            "DDR_unifie.csv",
            "text/csv"
        )

        st.subheader("Scatter : DDR vs DDR-V")
        chart = alt.Chart(df_ddr).mark_circle(size=80).encode(
            x=alt.X('DDR', title='DDR'),
            y=alt.Y('DDR-V', title='DDR-V (volume par match)'),
            color=alt.Color('Nom', title='Joueur'),
            tooltip=['Prénom','Nom','MIN','DDR','DDR%','DDR-V']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
