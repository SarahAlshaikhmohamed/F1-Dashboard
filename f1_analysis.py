<<<<<<< HEAD
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
=======
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="F1 Winners Analysis",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("🏎️ Formula 1 Winners Analysis (1950-2025)")

# Load dataset
@st.cache_data
def load_data():
    
    try:
        # Try to load from URL first
        url = "https://github.com/SarahAlshaikhmohamed/F1-Dashboard"
        df = pd.read_csv(url)
    except:
        # Fallback to local file
        try:
            df = pd.read_csv("f1_winners.csv")
        except:
            # Create sample data if file not found
            st.warning("Using sample data as the dataset file was not found.")
            years = list(range(1950, 2024))
            np.random.seed(42)
            data = {
                'Year': years,
                'Winner': np.random.choice(['Lewis Hamilton', 'Michael Schumacher', 'Juan Manuel Fangio', 
                                           'Ayrton Senna', 'Alain Prost', 'Sebastian Vettel', 
                                           'Fernando Alonso', 'Max Verstappen'], len(years)),
                'Nationality': np.random.choice(['British', 'German', 'Argentine', 'Brazilian', 
                                                'French', 'Dutch', 'Spanish'], len(years)),
                'Team': np.random.choice(['Mercedes', 'Ferrari', 'McLaren', 'Red Bull', 
                                         'Williams', 'Lotus', 'Brabham'], len(years)),
                'Points': np.random.randint(30, 120, len(years)),
                'Races': np.random.randint(10, 20, len(years)),
                'Wins': np.random.randint(5, 15, len(years)),
                'Podiums': np.random.randint(8, 18, len(years))
            }
            df = pd.DataFrame(data)
    return df

df = load_data()

# Sidebar
with st.sidebar:
    st.header("About the Dataset")
    st.markdown("""
    **Formula 1 Winners Dataset (1950-2025)**
    
    This dataset contains information about all Formula 1 World Championship winners
    from 1950 to the most recent season.
    
    **Columns:**
    - Year: Season year
    - Winner: Driver's name
    - Nationality: Driver's nationality
    - Team: Constructor team
    - Points: Championship points
    - Races: Number of races in the season
    - Wins: Number of wins by the champion
    - Podiums: Number of podium finishes
    """)
    
    st.header("Filters")
    
    # Year range filter
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    year_range = st.slider(
        "Select Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Nationality filter
    nationalities = st.multiselect(
        "Select Nationalities:",
        options=df['Nationality'].unique(),
        default=df['Nationality'].unique()
    )
    
    # Team filter
    teams = st.multiselect(
        "Select Teams:",
        options=df['Team'].unique(),
        default=df['Team'].unique()
    )
    
    # Wins filter
    min_wins = int(df['Wins'].min())
    max_wins = int(df['Wins'].max())
    wins_range = st.slider(
        "Select Wins Range:",
        min_value=min_wins,
        max_value=max_wins,
        value=(min_wins, max_wins)
    )

# Apply filters
filtered_df = df[
    (df['Year'] >= year_range[0]) & 
    (df['Year'] <= year_range[1]) &
    (df['Nationality'].isin(nationalities)) &
    (df['Team'].isin(teams)) &
    (df['Wins'] >= wins_range[0]) & 
    (df['Wins'] <= wins_range[1])
]

# Main area
st.header("Summary Statistics")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Champions", filtered_df['Winner'].nunique())
col2.metric("Most Championships", filtered_df['Winner'].value_counts().index[0])
col3.metric("Highest Points", f"{filtered_df['Points'].max()}")
col4.metric("Most Wins in Season", f"{filtered_df['Wins'].max()}")

# Show detailed statistics
if st.checkbox("Show detailed statistics"):
    st.subheader("Dataset Statistics")
    st.write(filtered_df.describe())

st.header("Interactive Visualizations")

# Create tabs for different visualizations
viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Champions Analysis", "Team Performance", "Historical Trends"])

with viz_tab1:
    st.subheader("Champions by Nationality")
    nationality_counts = filtered_df['Nationality'].value_counts()
    fig = px.pie(values=nationality_counts.values, names=nationality_counts.index, 
                 title="Distribution of Champions by Nationality")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top Champions")
    champion_wins = filtered_df['Winner'].value_counts().head(10)
    fig = px.bar(x=champion_wins.values, y=champion_wins.index, orientation='h',
                 title="Drivers with Most Championships", 
                 labels={'x': 'Number of Championships', 'y': 'Driver'})
    st.plotly_chart(fig, use_container_width=True)

with viz_tab2:
    st.subheader("Championships by Team")
    team_wins = filtered_df['Team'].value_counts().head(10)
    fig = px.bar(x=team_wins.values, y=team_wins.index, orientation='h',
                 title="Constructors with Most Championships", 
                 labels={'x': 'Number of Championships', 'y': 'Team'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Average Points by Team")
    team_avg_points = filtered_df.groupby('Team')['Points'].mean().sort_values(ascending=False).head(10)
    fig = px.bar(x=team_avg_points.values, y=team_avg_points.index, orientation='h',
                 title="Average Points per Championship by Team", 
                 labels={'x': 'Average Points', 'y': 'Team'})
    st.plotly_chart(fig, use_container_width=True)

with viz_tab3:
    st.subheader("Points Evolution Over Time")
    fig = px.line(filtered_df, x='Year', y='Points', 
                 title="Championship Points Over Time",
                 labels={'Points': 'Championship Points', 'Year': 'Season'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Wins vs Podiums")
    fig = px.scatter(filtered_df, x='Wins', y='Podiums', color='Team',
                 size='Points', hover_data=['Winner', 'Year'],
                 title="Relationship Between Wins and Podiums")
    st.plotly_chart(fig, use_container_width=True)

st.header("Data Preview")
st.dataframe(filtered_df.reset_index(drop=True))

# Show raw data option
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(df)

# Add some interesting facts
st.header("Interesting Facts")
col1, col2, col3 = st.columns(3)

with col1:
    most_championships = filtered_df['Winner'].value_counts().index[0]
    count = filtered_df['Winner'].value_counts().iloc[0]
    st.info(f"🏆 {most_championships} has the most championships ({count})")

with col2:
    highest_points = filtered_df.loc[filtered_df['Points'].idxmax()]
    st.info(f"📈 Highest points in a season: {highest_points['Points']} by {highest_points['Winner']} in {int(highest_points['Year'])}")

with col3:
    most_wins = filtered_df.loc[filtered_df['Wins'].idxmax()]
    st.info(f"🚀 Most wins in a season: {most_wins['Wins']} by {most_wins['Winner']} in {int(most_wins['Year'])}")

# Footer
st.markdown("---")

st.markdown("**Dataset Source:** [Kaggle - Formula 1 Winners 1950-2025](https://www.kaggle.com/datasets/julianbloise/winners-formula-1-1950-to-2025)")
>>>>>>> dd59610840816e505333d9eb628870482a089b26
