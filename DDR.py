import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leagueleaders
import altair as alt

# -----------------------------
# Pondérations volume
# -----------------------------
W_STEAL = 1.0
W_BLOCK = 0.9
W_DEFLECTION = 0.6
W_FOUL = 1.2  # poids du volume de fautes

# -----------------------------
# Chargement OppPtsPoss + 4 facteurs depuis Excel
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
                df_opp[col].astype(str)
                .str.replace('%','', regex=False)
                .str.replace(',','.', regex=False)
            )
            df_opp[col] = pd.to_numeric(df_opp[col], errors='coerce')

    # Convertir % en décimales
    for col in ['STL%','BLK%','PF%','OPP_EFG%','OPP_TOV%','OPP_ORB%','OPP_FTR']:
        if col in df_opp.columns:
            df_opp[col] = df_opp[col] / 100.0

    # Harmonisation des noms
    df_opp['PLAYER'] = df_opp['PLAYER'].str.strip().str.upper()
    df_opp = df_opp.drop_duplicates(subset='PLAYER', keep='first')
    return df_opp

# -----------------------------
# Calcul DDR unique
# -----------------------------
def compute_ddr(df_indiv, df_opp):
    df_indiv['PLAYER'] = df_indiv['PLAYER'].str.strip().str.upper()
    df = pd.merge(df_indiv, df_opp, on='PLAYER', how='left')

    # Remplissage des NaN
    for col in ['STL','BLK','PF','MIN','GP','DEFLECTIONS','OPPPTSPOSS',
                'STL%','BLK%','PF%','OPP_EFG%','OPP_TOV%','OPP_ORB%','OPP_FTR']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Volumes
    df['VolPos'] = W_STEAL * df['STL'] + W_BLOCK * df['BLK'] + W_DEFLECTION * df['DEFLECTIONS']
    df['VolNeg'] = W_FOUL * df['PF']

    # Noyau signé (log-ratio)
    df['core'] = np.log((df['VolPos'] + 1.0) / (df['VolNeg'] + 1.0))

    # Contexte individuel (robuste)
    def robust_z(series):
        med = series.median()
        iqr = series.quantile(0.75) - series.quantile(0.25)
        return (series - med) / (iqr if iqr > 0 else 1e-6)

    z_stl = robust_z(df['STL%'])
    z_blk = robust_z(df['BLK%'])
    z_pf  = robust_z(df['PF%'])

    ind_raw = 0.8 * z_stl + 0.6 * z_blk - 0.7 * z_pf
    df['ContextE'] = 1.0 + 0.2 * np.tanh(ind_raw)

    # Contexte collectif
    team_4f = (
        1.1 * (1.0 - df['OPP_EFG%']) +
        1.3 * df['OPP_TOV%'] +
        1.0 * (1.0 - df['OPP_ORB%']) +
        1.1 * (1.0 - df['OPP_FTR'])
    )
    z_opp_ppp = robust_z(df['OPPPTSPOSS'])
    team_raw = team_4f - 0.3 * z_opp_ppp
    df['ContextTeam'] = 1.0 + 0.15 * np.tanh(team_raw)

    # Calibration globale
    core_med = df['core'].median()
    core_iqr = df['core'].quantile(0.75) - df['core'].quantile(0.25)
    core_norm = (df['core'] - core_med) / (core_iqr if core_iqr > 0 else 1e-6)

    # Présentation
    df['Prénom'] = df['PLAYER'].str.split().str[0].str.capitalize()
    df['Nom'] = df['PLAYER'].str.split().str[1:].str.join(' ').str.capitalize()
    df['Rank DDR'] = df['DDR'].rank(ascending=False, method='min').fillna(0).astype(int)

    return df[['Prénom','Nom','TEAM','MIN','DDR','Rank DDR']].sort_values('DDR', ascending=False)

# -----------------------------
# Interface Streamlit
# -----------------------------
st.title("Defensive Disruption Rate (DDR) — Saison sélectionnable, DDR unique")

st.info("""
- **DDR unique**: log-ratio VolPos vs VolNeg corrigé par contexte individuel (% STL/BLK/PF) et collectif (4 facteurs + opp pts/poss).
- **Calibration**: centrage par médiane + échelle IQR, compression par tanh pour lisibilité.
- Échelle cible: environ -3 à +10.
""")

season = st.selectbox(
    "Choisir la saison NBA",
    options=["2024-25", "2025-26"],
    index=1
)

# ✅ Correction du slider
min_threshold = st.slider("Minutes minimum", 0, 2000, 500, step=50)
selected_team = st.text_input("Équipe (laisser vide pour toutes)", value="")

@st.cache_data
def fetch_league_leaders(season="2025-26"):
    ll = leagueleaders.LeagueLeaders(season=season, season_type_all_star="Regular Season")
    df = ll.get_data_frames()[0]
    return df[['PLAYER','TEAM','GP','MIN','STL','BLK','PF']].copy()

if st.button("Générer DDR"):
    with st.spinner("Chargement des données..."):
        df_indiv = fetch_league_leaders(season)
        df_opp = fetch_opp_excel(
            "opp_pts_poss25-26.xlsx" if season == "2025-26" else "opp_pts_poss24_25.xlsx"
        )
        df_ddr = compute_ddr(df_indiv, df_opp)

        if 'TEAM' in df_ddr.columns and selected_team.strip():
            df_ddr = df_ddr[df_ddr['TEAM'] == selected_team]
        df_ddr = df_ddr[df_ddr['MIN'] >= min_threshold]

        st.subheader(f"Classement DDR ({season})")
        st.dataframe(df_ddr)

        st.download_button(
            f"Télécharger le classement complet ({season})",
            df_ddr.to_csv(index=False).encode('utf-8'),
            f"DDR_{season}.csv",
            "text/csv"
        )

        st.subheader("Scatter : DDR vs Minutes")
        chart = alt.Chart(df_ddr).mark_circle(size=80).encode(
            x=alt.X('DDR', title='DDR'),
            y=alt.Y('MIN', title='Minutes'),
            color=alt.Color('Nom', title='Joueur'),
            tooltip=['Prénom','Nom','TEAM','MIN','DDR','Rank DDR']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Histogramme de la distribution des DDR")
        hist = alt.Chart(df_ddr).mark_bar().encode(
            alt.X("DDR", bin=alt.Bin(maxbins=30), title="DDR"),
            alt.Y("count()", title="Nombre de joueurs"),
            tooltip=["count()"]
        ).properties(width=600, height=400)
        st.altair_chart(hist, use_container_width=True)
