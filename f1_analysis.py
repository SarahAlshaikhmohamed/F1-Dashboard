import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import requests
from datetime import datetime
import os

# Set page configuration
st.set_page_config(
    page_title="F1 Winners Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("/Users/abdul/Desktop/project g6/F1-Dashboard/Dataset/winners_f1_cleaned.csv")
    except:
        st.error("Dataset not found. Please place 'Dataset\winners_f1_cleaned.csv' in the same folder.")
        return pd.DataFrame()

    # Standardize column names
    df = df.rename(columns={
        "Winner-Name": "Winner",
        "Grand-Prix": "GrandPrix"
    })

    return df

df = load_data()

if df.empty:
    st.stop()

# Sidebar
with st.sidebar:
    st.sidebar.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

with st.sidebar:
    st.header("About the Dataset")
    st.markdown("""
    **Formula 1 Winners Dataset (1950-2025)**
    
    This dataset contains information about all Formula 1 World Championship winners
    from 1950 to the most recent season.
    """)
    st.header("Filters")

    # Year range
    min_year, max_year = int(df["Year"].min()), int(df["Year"].max())
    year_range = st.slider(
        "Select Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    # Winner filter
    winners = st.multiselect(
        "Select Winners:",
        options=df["Winner"].unique(),
        default=df["Winner"].unique()
    )

    # Team filter
    teams = st.multiselect(
        "Select Teams:",
        options=df["Team"].unique(),
        default=df["Team"].unique()
    )

    # Continent filter
    continents = st.multiselect(
        "Select Continents:",
        options=df["Continent"].unique(),
        default=df["Continent"].unique()
    )

# Apply filters
filtered_df = df[
    (df["Year"] >= year_range[0]) &
    (df["Year"] <= year_range[1]) &
    (df["Winner"].isin(winners)) &
    (df["Team"].isin(teams)) &
    (df["Continent"].isin(continents))
]

# Main area
st.title("🏎️ Formula 1 Dashboard")

st.header("Summary Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Races", len(filtered_df))
col2.metric("Unique Winners", filtered_df["Winner"].nunique())
col3.metric("Unique Teams", filtered_df["Team"].nunique())

# Top winner
if not filtered_df.empty:
    top_winner = filtered_df["Winner"].value_counts().idxmax()
    top_wins = filtered_df["Winner"].value_counts().max()
    st.success(f"🏆 {top_winner} has the most race wins in this selection ({top_wins})")

if st.checkbox("Show detailed statistics"):
    st.subheader("Dataset Statistics")
    try:
        st.write(filtered_df.describe(include='all'))
    except Exception as e:
        st.write("Could not produce descriptive statistics:", e)

st.header("Visualizations")
tab1, tab2, tab3 = st.tabs(["Winners", "Teams", "Race Trends"])

with tab1:
    st.subheader("Most Successful Winners")
    winner_counts = filtered_df["Winner"].value_counts().head(10)
    fig = px.bar(
        x=winner_counts.values,
        y=winner_counts.index,
        orientation="h",
        labels={"x": "Number of Wins", "y": "Winner"},
        title="Top Winning Drivers"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Team Performance")
    team_counts = filtered_df["Team"].value_counts().head(10)
    fig = px.bar(
        x=team_counts.values,
        y=team_counts.index,
        orientation="h",
        labels={"x": "Number of Wins", "y": "Team"},
        title="Top Winning Teams"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Race Trends Over Time")
    races_per_year = filtered_df.groupby("Year")["GrandPrix"].count()
    fig = px.line(
        x=races_per_year.index,
        y=races_per_year.values,
        labels={"x": "Year", "y": "Number of Races"},
        title="Number of Races per Year"
    )
    st.plotly_chart(fig, use_container_width=True)

# FASTAPI PREDICTION SECTION
st.header("Race Time Prediction")
col1, col2, col3, col4 = st.columns(4)

with col1:
    continent = st.selectbox("Continent", ["Europe", "Asia", "North America", "South America", "Australia", "Africa"])

with col2:
    team_for_pred = st.selectbox("Team", df['Team'].dropna().unique() if 'Team' in df.columns else ["Mercedes"])

with col3:
    laps = st.slider("Laps", min_value=10, max_value=200, value=70)

with col4:
    year_pred = st.slider("Year", min_value=2026, max_value=2030, value=2026)

if st.button("Predict"):
    url = "http://127.0.0.1:8000/predict"
    payload = {
        "continent": continent,
        "team": team_for_pred,
        "laps": int(laps),
        "year": int(year_pred)
    }
    try:
        # send JSON body (POST)
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        st.success(f"Predicted Total Race Time in Seconds: {result.get('predicted_time_seconds', 0):.1f} seconds")
        st.info(f"Predicted Total Time: {result.get('predicted_time_hhmmss', '00:00:00')}")
    except Exception as e:
        st.error(f"Error connecting to prediction API: {e}")
        st.info("Make sure your FastAPI server is running on http://127.0.0.1:8000")

st.header("Data Preview")
st.dataframe(filtered_df.reset_index(drop=True))

if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(df)

st.header("Interesting Facts")
col1, col2, col3 = st.columns(3)

with col1:
    try:
        most_championships = filtered_df['Winner'].value_counts().idxmax()
        count = int(filtered_df['Winner'].value_counts().max())
        st.info(f"🏆 {most_championships} has the most championships ({count})")
    except Exception:
        st.info("🏆 No champion data for selected filters")

with col2:
    try:
        highest_row = filtered_df.loc[filtered_df['Time'].idxmax()]
        st.info(f"⏱️ Longest race time: {highest_row['Time']} by {highest_row['Winner']} in {int(highest_row['Year'])}")
    except Exception:
        st.info("⏱️ No race time data for selected filters")
with col3:
    try:
        most_laps_row = filtered_df.loc[filtered_df['Laps'].idxmax()]
        st.info(f"🔄 Most laps in a race: {int(most_laps_row['Laps'])} by {most_laps_row['Winner']} in {int(most_laps_row['Year'])}")
    except Exception:
        st.info("🔄 No laps data for selected filters")

st.markdown("---")

st.markdown("**Dataset Source:** [Kaggle - Formula 1 Winners 1950-2025](https://www.kaggle.com/datasets/julianbloise/winners-formula-1-1950-to-2025)")
