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
W_DEFLECTION = 1.0

# -----------------------------
# Chargement OppPtsPoss + % + deflections depuis Excel
# -----------------------------
def fetch_opp_excel(path="opp_pts_poss24_25.xlsx"):
    df_opp = pd.read_excel(path)
    df_opp.columns = df_opp.columns.str.strip().str.upper()

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

    # Harmonisation des noms pour éviter les doublons
    df_opp['PLAYER'] = df_opp['PLAYER'].str.strip().str.upper()

    # Suppression des doublons (on garde la première occurrence)
    df_opp = df_opp.drop_duplicates(subset='PLAYER', keep='first')

    return df_opp

# -----------------------------
# Calcul DDR unifié
# -----------------------------
def compute_ddr(df_indiv, df_opp):
    # Harmonisation des noms côté NBA API
    df_indiv['PLAYER'] = df_indiv['PLAYER'].str.strip().str.upper()

    df = pd.merge(df_indiv, df_opp, on='PLAYER', how='left')

    for col in ['STL','BLK','PF','MIN','GP','OPPPTSPOSS','STL%','BLK%','PF%','DEFLECTIONS']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # DDR-E (efficacité pondérée, mis à l'échelle)
    df['DDR-E'] = (
        W_STEAL * df['STL%'] +
        W_BLOCK * df['BLK%'] +
        W_FOUL  * df['PF%']
    )
    df['DDR-E'] = df['DDR-E'] * 1000

    # Volume positif et négatif pondérés
    df['VolPos'] = (
        W_STEAL * df['STL'] +
        W_BLOCK * df['BLK'] +
        W_DEFLECTION * df['DEFLECTIONS']
    )
    df['VolNeg'] = abs(W_FOUL) * df['PF']

    # Contexte individuel (lié à DDR-E)
    df['ContextE'] = np.where(df['DDR-E'] > 0, 1.1, 0.9)

    # Contexte collectif (lié à OppPtsPoss)
    df['ContextTeam'] = np.where(df['OPPPTSPOSS'] < 100, 1.1, 0.9)

    # DDR final = rapport VolPos / VolNeg × double contexte
    df['DDR'] = np.where(df['VolNeg'] != 0,
                         (df['VolPos'] / df['VolNeg']) * df['ContextE'] * df['ContextTeam'],
                         np.nan)

    df['Prénom'] = df['PLAYER'].str.split().str[0].str.capitalize()
    df['Nom'] = df['PLAYER'].str.split().str[1:].str.join(' ').str.capitalize()

    # Ajout des colonnes de classement avec gestion des NaN
    df['Rank DDR-E'] = df['DDR-E'].rank(ascending=False, method='min')
    df['Rank DDR-E'] = df['Rank DDR-E'].fillna(0).astype(int)

    df['Rank DDR'] = df['DDR'].rank(ascending=False, method='min')
    df['Rank DDR'] = df['Rank DDR'].fillna(0).astype(int)

    # Inclure TEAM + ranks
    df_final = df[['Prénom','Nom','TEAM','MIN','DDR-E','Rank DDR-E','DDR','Rank DDR']]
    return df_final.sort_values('DDR', ascending=False)

# -----------------------------
# Interface Streamlit
# -----------------------------
st.title("Defensive Disruption Rate (DDR) -- Powered by Pano")

# Encadré explicatif coloré
st.info("""
🧾 **Comprendre le DDR et le DDR‑E**

- **DDR‑E (Efficiency)**  
  → Mesure l’efficacité défensive pondérée par possession.  
  → Score purement individuel : le DDR-E met en lumière la qualité des actions défensives.  

- **DDR (Final)**  
  → Mesure le rapport volume positif/négatif corrigé par double contexte.  
  → Score global : valorise les profils **propres et actifs**.  

**Lecture rapide :**  
- DDR‑E ↑ + DDR ↑ → défenseur efficace et propre (profil élite).  
- DDR‑E ↑ + DDR ↓ → défenseur efficace mais trop de fautes (profil agressif).  
- DDR‑E ↓ + DDR ↑ → joueur actif qui perturbe sans être élite (profil opportuniste).  
- DDR‑E ↓ + DDR ↓ → défenseur peu impactant et pénalisant (profil fragile).
""")

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
        df_opp = fetch_opp_excel("opp_pts_poss24_25.xlsx")

        df_ddr = compute_ddr(df_indiv, df_opp)

        # Filtre par équipe sécurisé
        if 'TEAM' in df_ddr.columns and selected_team.strip():
            df_ddr = df_ddr[df_ddr['TEAM'] == selected_team]
        df_ddr = df_ddr[df_ddr['MIN'] >= min_threshold]

        st.subheader("Classement DDR unifié (double contexte + rangs)")
        st.dataframe(
            df_ddr,
            column_config={
                "DDR": st.column_config.NumberColumn(
                    "DDR",
                    help="Score final = (VolPos/VolNeg) × ContextE × ContextTeam",
                    format="%.2f",
                    min_value=df_ddr["DDR"].min(),
                    max_value=df_ddr["DDR"].max()
                ),
                "DDR-E": st.column_config.NumberColumn(
                    "DDR-E",
                    help="Efficacité défensive pondérée (échelle -100 à +100, fautes négatives)",
                    format="%.1f",
                    min_value=df_ddr["DDR-E"].min(),
                    max_value=df_ddr["DDR-E"].max()
                ),
                "Rank DDR": st.column_config.NumberColumn(
                    "Rank DDR",
                    help="Classement par DDR final",
                    format="%d"
                ),
                "Rank DDR-E": st.column_config.NumberColumn(
                    "Rank DDR-E",
                    help="Classement par efficacité pondérée DDR-E",
                    format="%d"
                )
            }
        )

        st.download_button(
            "Télécharger le classement complet",
            df_ddr.to_csv(index=False).encode('utf-8'),
            "DDR_double_contexte_rangs.csv",
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
