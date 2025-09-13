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
    # Load the dataset - replace with your actual file path
    # You can upload the file directly or provide a URL
    try:
        # Try to load from URL first
        url = "https://github.com/your-username/f1-data/raw/main/f1_winners.csv"  # Replace with your actual URL
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