import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leagueleaders
import altair as alt

# -----------------------------
# Weights and blend
# -----------------------------
W_STEAL = 1.8
W_BLOCK = 1.4
W_FOUL = -1.5
W_DEFLECTION = 1.2
ALPHA_BLEND = 0.5  # 0..1  (0 = only per36 volume, 1 = only rate)

def safe_per36(value, minutes):
    return (value / minutes) * 36 if minutes and minutes > 0 else np.nan

# -----------------------------
# NBA API totals
# -----------------------------
@st.cache_data
def fetch_league_leaders(season="2024-25"):
    ll = leagueleaders.LeagueLeaders(season=season, season_type_all_star="Regular Season")
    df = ll.get_data_frames()[0]
    return df[['PLAYER','TEAM','GP','MIN','STL','BLK','PF']].copy()

# -----------------------------
# Excel: OppPtsPoss + % + deflections
# -----------------------------
@st.cache_data
def fetch_opp_excel(path="opp_pts_poss.xlsx"):
    df_opp = pd.read_excel(path)
    df_opp.columns = df_opp.columns.str.strip().str.upper()

    required = ['PLAYER','OPPPTSPOSS','STL%','BLK%','PF%','DEFLECTIONS']
    missing = [c for c in required if c not in df_opp.columns]
    if missing:
        st.error(f"Colonnes manquantes dans Excel: {missing}. Colonnes trouvées: {df_opp.columns.tolist()}")
        # Create empty safe frame
        for c in missing:
            df_opp[c] = 0.0
        if 'PLAYER' not in df_opp.columns:
            df_opp['PLAYER'] = ""
    return df_opp

# -----------------------------
# Unified DDR compute
# -----------------------------
def compute_ddr(df_indiv, df_opp):
    df = pd.merge(df_indiv, df_opp, on='PLAYER', how='left')

    # Fill missing
    for col in ['STL','BLK','PF','MIN','OPPPTSPOSS','STL%','BLK%','PF%','DEFLECTIONS']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Per-36 totals
    df['STL36'] = df.apply(lambda r: safe_per36(r['STL'], r['MIN']), axis=1)
    df['BLK36'] = df.apply(lambda r: safe_per36(r['BLK'], r['MIN']), axis=1)
    df['PF36']  = df.apply(lambda r: safe_per36(r['PF'],  r['MIN']), axis=1)

    # Rate component (percentages)
    df['DDR_rate'] = (
        W_STEAL * df['STL%'] +
        W_BLOCK * df['BLK%'] +
        W_FOUL  * df['PF%']
    )

    # Volume component (per-36 + deflections raw per-game)
    df['DDR_per36'] = (
        W_STEAL * df['STL36'] +
        W_BLOCK * df['BLK36'] +
        W_FOUL  * df['PF36']  +
        W_DEFLECTION * df['DEFLECTIONS']
    )

    # Blend
    df['DDR_blend'] = ALPHA_BLEND * df['DDR_rate'] + (1 - ALPHA_BLEND) * df['DDR_per36']

    # Context factor
    df['OppFactor'] = 1.3 - (df['OPPPTSPOSS'] / 100.0)

    # Final
    df['DDR_final'] = df['DDR_blend'] * df['OppFactor']

    # Name split
    df['Prénom'] = df['PLAYER'].str.split().str[0]
    df['Nom'] = df['PLAYER'].str.split().str[1:].str.join(' ')

    # Output
    cols = [
        'Prénom','Nom','TEAM','GP','MIN',
        'STL','BLK','PF','STL36','BLK36','PF36',
        'STL%','BLK%','PF%','DEFLECTIONS',
        'DDR_rate','DDR_per36','OppFactor','DDR_blend','DDR_final'
    ]
    return df[cols].sort_values('DDR_final', ascending=False)

# -----------------------------
# App
# -----------------------------
st.title("Defensive Disruption Rate (DDR) by Pano — Unified")

season = st.text_input("Saison NBA API (ex: 2024-25)", value="2024-25")
min_threshold = st.slider("Minutes minimum", 0, 2000, 500, 50)  # start simple: default 500
selected_team = st.text_input("Équipe (laisser vide pour toutes)", value="")
alpha_ui = st.slider("Mix taux vs volume (alpha)", 0.0, 1.0, ALPHA_BLEND, 0.05)

if st.button("Générer DDR"):
    with st.spinner("Chargement des données..."):
        df_indiv = fetch_league_leaders(season)
        df_opp = fetch_opp_excel("opp_pts_poss.xlsx")

        # Override alpha from UI
        global ALPHA_BLEND
        ALPHA_BLEND = alpha_ui

        df_ddr = compute_ddr(df_indiv, df_opp)

        # Filters
        if selected_team.strip():
            df_ddr = df_ddr[df_ddr['TEAM'] == selected_team]
        df_ddr = df_ddr[df_ddr['MIN'] >= min_threshold]

        st.subheader("Classement DDR unifié")
        st.dataframe(df_ddr)

        st.download_button(
            "Télécharger le classement complet",
            df_ddr.to_csv(index=False).encode('utf-8'),
            "DDR_unifie.csv",
            "text/csv"
        )

        st.subheader("Scatter : DDR_final vs DDR_rate")
        chart = alt.Chart(df_ddr).mark_circle(size=80).encode(
            x=alt.X('DDR_final', title='DDR Final'),
            y=alt.Y('DDR_rate', title='DDR (taux)'),
            color=alt.Color('TEAM', title='Équipe'),
            tooltip=['Prénom','Nom','TEAM','MIN','DDR_final','DDR_rate','DDR_per36','OppFactor']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
