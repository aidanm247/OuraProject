"""
Sleep, Pain & Perfor# Page config
st.set_page_config(
    page_title="Sleep, Pain & Performance Analytics",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed"
)nalytics Dashboard

An explorative case study analyzing the relationships between:
- Sleep metrics (total sleep, REM, deep sleep)
- Pain and soreness levels
- Basketball performance metrics

This app combines Oura API data with manual game logs to uncover patterns
and generate data-driven insights about recovery and performance.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv('.env.local')

# Page config
st.set_page_config(
    page_title="Sleep, Pain & Performance Analytics",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #2c3e50;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #555;
        font-weight: 500;
    }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    h1 {
        color: #2c3e50;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def get_oura_token():
    """Get Oura personal access token from environment"""
    #return os.getenv("OURA_PERSONAL_ACCESS_TOKEN") or os.getenv("OURA_TOKEN")
    return "SNW6SBMYGKH34XSEIYNFDJNJTA7OZRQN"


def fetch_sleep_data(token, start_date, end_date):
    """
    Fetch sleep data from Oura API for the specified date range
    """
    url = "https://api.ouraring.com/v2/usercollection/sleep"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d")
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from Oura API: {e}")
        return None


def format_duration(seconds):
    """Convert seconds to hours and minutes format"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours}h {minutes}m"


def load_game_data(csv_path='data/game_data_log.csv'):
    """Load game performance data from CSV"""
    try:
        df = pd.read_csv(csv_path)
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Parse date column (format is MM/DD/YYYY)
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
       
        
        # Parse sleep duration from "H:MM" format to seconds
        def parse_sleep_time(time_str):
            if pd.isna(time_str):
                return 0
            try:
                time_str = str(time_str).strip()
                parts = time_str.split(':')
                hours = int(parts[0])
                minutes = int(parts[1]) if len(parts) > 1 else 0
                return hours * 3600 + minutes * 60
            except Exception as e:
                return 0
        
        df['total_sleep_duration'] = df['total_sleep'].apply(parse_sleep_time)
        df['rem_sleep_duration'] = df['rem_sleep'].apply(parse_sleep_time)
        df['deep_sleep_duration'] = df['deep_sleep'].apply(parse_sleep_time)
        df['light_sleep_duration'] = df['light_sleep'].apply(parse_sleep_time)
        df['awake_time_duration'] = df['awake_time'].apply(parse_sleep_time)
        
        # Calculate shooting percentages
        df['fg_pct'] = (df['fgm'] / df['fg_attempted'] * 100).fillna(0).round(1)
        df['threept_pct'] = (df['threeptm'] / df['threept_attempted'] * 100).fillna(0).round(1)
        df['ft_pct'] = (df['ftm'] / df['ft_attempted'] * 100).fillna(0).round(1)
        df['total_rebs'] = df['orebs'] + df['drebs']
        
        return df
    except Exception as e:
        st.error(f"Error loading game data: {e}")
        import traceback
        st.error(traceback.format_exc())
        raise


def merge_sleep_and_game_data(sleep_df, game_df):
    """Merge sleep data with game performance data by date"""
    if sleep_df is None or game_df is None:
        return None
    
    # Ensure both have date columns
    if 'day' in sleep_df.columns:
        sleep_df['date'] = pd.to_datetime(sleep_df['day'])
    
    # Merge on date
    merged = pd.merge(
        game_df,
        sleep_df,
        on='date',
        how='left',
        suffixes=('_game', '_sleep')
    )
    
    return merged


def calculate_correlations(df, x_col, y_cols):
    """Calculate correlation coefficients between variables using pandas"""
    correlations = {}
    for y_col in y_cols:
        # Filter out NaN values
        mask = df[[x_col, y_col]].notna().all(axis=1)
        if mask.sum() > 1:  # Need at least 2 points
            # Use pandas correlation (Pearson by default)
            corr = df.loc[mask, [x_col, y_col]].corr().iloc[0, 1]
            correlations[y_col] = {'correlation': corr}
    return correlations


def render_metrics(record):
    """Display key sleep metrics in columns"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sleep", format_duration(record['total_sleep_duration']))
        st.metric("Deep Sleep", format_duration(record['deep_sleep_duration']))
    
    with col2:
        st.metric("Sleep Efficiency", f"{record['efficiency']}%")
        st.metric("REM Sleep", format_duration(record['rem_sleep_duration']))
    
    with col3:
        st.metric("Light Sleep", format_duration(record['light_sleep_duration']))
        st.metric("Awake Time", format_duration(record['awake_time']))
    
    with col4:
        st.metric("Avg Heart Rate", f"{round(record['average_heart_rate'])} bpm")
        st.metric("Readiness Score", record['readiness']['score'])


def plot_sleep_stages(record):
    """Create sleep stages timeline chart"""
    phase_map = {'1': 'Deep', '2': 'Light', '3': 'REM', '4': 'Awake'}
    color_map = {'Deep': '#2c3e50', 'Light': '#5a9fd4', 'REM': '#95a5a6', 'Awake': '#e74c3c'}
    
    phases = [phase_map.get(p, 'Unknown') for p in record['sleep_phase_5_min']]
    start_time = datetime.fromisoformat(record['bedtime_start'].replace('Z', '+00:00'))
    
    times = [start_time + timedelta(minutes=i*5) for i in range(len(phases))]
    time_labels = [t.strftime('%I:%M %p') for t in times]
    
    # Create categorical plot with better visibility
    fig = go.Figure()
    
    # Create area for each phase
    for phase_name in ['Deep', 'Light', 'REM', 'Awake']:
        phase_y = [phase if phase == phase_name else None for phase in phases]
        fig.add_trace(go.Scatter(
            x=time_labels,
            y=phase_y,
            mode='markers',
            marker=dict(
                color=color_map[phase_name],
                size=10,
                symbol='square'
            ),
            name=phase_name,
            hovertemplate='<b>%{y}</b><br>Time: %{x}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text="Sleep Stages Timeline", font=dict(size=20, color='#000', family='Arial Black')),
        xaxis=dict(
            title=dict(text="Time", font=dict(size=14, color='#000', family='Arial')),
            tickangle=-45,
            gridcolor='#e8e8e8',
            showgrid=True,
            linecolor='#333',
            linewidth=1,
            tickfont=dict(size=12, color='#000')
        ),
        yaxis=dict(
            title=dict(text="Sleep Stage", font=dict(size=14, color='#000', family='Arial')),
            categoryorder='array',
            categoryarray=['Awake', 'REM', 'Light', 'Deep'],
            gridcolor='#e8e8e8',
            showgrid=True,
            linecolor='#333',
            linewidth=1,
            tickfont=dict(size=12, color='#000')
        ),
        height=450,
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=12, color='#000')
        ),
        font=dict(color='#000')
    )
    
    return fig


def plot_heart_rate_hrv(record):
    """Create dual-axis chart for heart rate and HRV"""
    hr_data = record['heart_rate']
    hrv_data = record['hrv']
    
    start_time = datetime.fromisoformat(hr_data['timestamp'].replace('Z', '+00:00'))
    times = [start_time + timedelta(seconds=i*hr_data['interval']) for i in range(len(hr_data['items']))]
    time_labels = [t.strftime('%I:%M %p') for t in times]
    
    fig = go.Figure()
    
    # Heart Rate trace
    fig.add_trace(go.Scatter(
        x=time_labels,
        y=hr_data['items'],
        mode='lines',
        name='Heart Rate (bpm)',
        line=dict(color='#5a9fd4', width=2),
        yaxis='y'
    ))
    
    # HRV trace
    fig.add_trace(go.Scatter(
        x=time_labels,
        y=hrv_data['items'],
        mode='lines',
        name='HRV (ms)',
        line=dict(color='#7b8a9c', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=dict(text="Heart Rate & Heart Rate Variability", font=dict(size=20, color='#000', family='Arial Black')),
        xaxis=dict(
            title=dict(text="Time", font=dict(size=14, color='#000', family='Arial')),
            tickangle=-45,
            gridcolor='#e8e8e8',
            showgrid=True,
            linecolor='#333',
            linewidth=1,
            tickfont=dict(size=12, color='#000')
        ),
        yaxis=dict(
            title=dict(text="Heart Rate (bpm)", font=dict(size=14, color='#000', family='Arial')),
            tickfont=dict(size=12, color='#000'),
            gridcolor='#e8e8e8',
            showgrid=True,
            linecolor='#333',
            linewidth=1
        ),
        yaxis2=dict(
            title=dict(text="HRV (ms)", font=dict(size=14, color='#000', family='Arial')),
            tickfont=dict(size=12, color='#000'),
            overlaying='y',
            side='right',
            gridcolor='#e8e8e8',
            linecolor='#333',
            linewidth=1
        ),
        height=450,
        hovermode='x unified',
        legend=dict(
            x=0.5,
            y=1.1,
            xanchor='center',
            orientation='h',
            font=dict(size=12, color='#000')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#000')
    )
    
    return fig


def plot_sleep_breakdown(record):
    """Create pie chart for sleep duration breakdown"""
    values = [
        record['deep_sleep_duration'] / 60,
        record['light_sleep_duration'] / 60,
        record['rem_sleep_duration'] / 60,
        record['awake_time'] / 60
    ]
    labels = ['Deep Sleep', 'Light Sleep', 'REM Sleep', 'Awake']
    colors = ['#2c3e50', '#5a9fd4', '#95a5a6', '#e74c3c']
    
    fig = go.Figure(data=[go.Pie(
        values=values,
        labels=labels,
        hole=0.4,
        marker=dict(
            colors=colors,
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=13, color='#000', family='Arial'),
        hovertemplate='<b>%{label}</b><br>%{value:.0f} minutes<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text="Sleep Duration Breakdown", font=dict(size=20, color='#000', family='Arial Black')),
        height=450,
        showlegend=True,
        legend=dict(
            x=0.5,
            y=-0.1,
            xanchor='center',
            orientation='h',
            font=dict(size=12, color='#000')
        ),
        paper_bgcolor='white',
        font=dict(color='#000'),
        annotations=[dict(
            text=format_duration(record['total_sleep_duration']),
            x=0.5, y=0.5,
            font=dict(size=26, color='#000', family='Arial Black'),
            showarrow=False
        )]
    )
    
    return fig


def plot_readiness(record):
    """Create horizontal bar chart for readiness contributors"""
    contributors = record['readiness']['contributors']
    
    # Format labels and filter out None values
    labels = []
    values = []
    for k, v in contributors.items():
        if v is not None:
            labels.append(k.replace('_', ' ').title())
            values.append(v)
    
    # Assign colors based on value ranges
    colors = []
    for v in values:
        if v >= 85:
            colors.append('#27ae60')  # Green for good
        elif v >= 70:
            colors.append('#5a9fd4')  # Blue for decent
        elif v >= 50:
            colors.append('#f39c12')  # Orange for fair
        else:
            colors.append('#e74c3c')  # Red for poor
    
    fig = go.Figure(data=[go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='#333', width=1)
        ),
        text=[f"{v}%" for v in values],
        textposition='outside',
        textfont=dict(size=13, color='#000', family='Arial'),
        hovertemplate='<b>%{y}</b><br>Score: %{x}/100<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text="Readiness Score Contributors", font=dict(size=20, color='#000', family='Arial Black')),
        xaxis=dict(
            title=dict(text="Score", font=dict(size=14, color='#000', family='Arial')),
            range=[0, 105],
            gridcolor='#e8e8e8',
            showgrid=True,
            linecolor='#333',
            linewidth=1,
            tickfont=dict(size=12, color='#000')
        ),
        yaxis=dict(
            gridcolor='#e8e8e8',
            showgrid=False,
            linecolor='#333',
            linewidth=1,
            tickfont=dict(size=12, color='#000')
        ),
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=200),
        font=dict(color='#000')
    )
    
    return fig


# ========== NEW PERFORMANCE ANALYTICS FUNCTIONS ==========

def plot_sleep_vs_performance(merged_df):
    """Scatter plots showing sleep metrics vs performance"""
    fig = go.Figure()
    
    # Total sleep vs Points
    fig.add_trace(go.Scatter(
        x=merged_df['total_sleep_duration'] / 3600,
        y=merged_df['pts'],
        mode='markers+text',
        marker=dict(size=12, color='#5a9fd4', line=dict(color='#000', width=1)),
        text=merged_df['date'].dt.strftime('%m/%d'),
        textposition='top center',
        textfont=dict(size=10, color='#000'),
        name='Points',
        hovertemplate='Sleep: %{x:.1f}h<br>Points: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Total Sleep vs Points Scored", font=dict(size=20, color='#000', family='Arial Black')),
        xaxis=dict(
            title=dict(text="Total Sleep (hours)", font=dict(size=14, color='#000', family='Arial')),
            gridcolor='#e8e8e8',
            showgrid=True,
            tickfont=dict(size=12, color='#000')
        ),
        yaxis=dict(
            title=dict(text="Points", font=dict(size=14, color='#000', family='Arial')),
            gridcolor='#e8e8e8',
            showgrid=True,
            tickfont=dict(size=12, color='#000')
        ),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#000')
    )
    
    return fig


def plot_shooting_efficiency_matrix(merged_df):
    """Heatmap-style visualization of shooting efficiency vs sleep and soreness"""
    fig = go.Figure()
    
    # FG% vs Sleep
    fig.add_trace(go.Scatter(
        x=merged_df['total_sleep_duration'] / 3600,
        y=merged_df['fg_pct'],
        mode='markers',
        marker=dict(
            size=merged_df['soreness'] * 10,  # Size based on soreness
            color=merged_df['pain'],  # Color based on pain
            colorscale='RdYlGn_r',  # Red = high pain, Green = low pain
            showscale=True,
            colorbar=dict(title='Pain Level', tickfont=dict(size=12, color='#000')),
            line=dict(color='#000', width=1)
        ),
        text=merged_df['date'].dt.strftime('%m/%d'),
        textposition='top center',
        textfont=dict(size=10, color='#000'),
        hovertemplate='Sleep: %{x:.1f}h<br>FG%%: %{y:.1f}%%<br>Pain: %{marker.color}<br>Soreness: %{marker.size:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="Shooting Efficiency vs Sleep & Recovery", font=dict(size=20, color='#000', family='Arial Black')),
        xaxis=dict(
            title=dict(text="Total Sleep (hours)", font=dict(size=14, color='#000', family='Arial')),
            gridcolor='#e8e8e8',
            showgrid=True,
            tickfont=dict(size=12, color='#000')
        ),
        yaxis=dict(
            title=dict(text="Field Goal %", font=dict(size=14, color='#000', family='Arial')),
            gridcolor='#e8e8e8',
            showgrid=True,
            tickfont=dict(size=12, color='#000')
        ),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#000'),
        annotations=[dict(
            text='Bubble size = Soreness level',
            xref='paper', yref='paper',
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=11, color='#555')
        )]
    )
    
    return fig


def plot_best_worst_comparison(merged_df):
    """Compare sleep before best and worst performances"""
    # Define performance score (weighted combination)
    merged_df['performance_score'] = (
        merged_df['pts'] * 1.0 +
        merged_df['total_rebs'] * 1.2 +
        merged_df['ast'] * 1.5 +
        merged_df['fg_pct'] * 0.3 +
        merged_df['plus_minus'] * 0.5
    )
    
    # Get best and worst games
    best_game = merged_df.nlargest(1, 'performance_score').iloc[0]
    worst_game = merged_df.nsmallest(1, 'performance_score').iloc[0]
    
    # Create comparison data
    categories = ['Total Sleep (h)', 'REM Sleep (min)', 'Deep Sleep (min)', 
                  'Sleep Score', 'Readiness', 'Pain', 'Soreness']
    
    best_values = [
        best_game['total_sleep_duration'] / 3600,
        best_game['rem_sleep_duration'] / 60,
        best_game['deep_sleep_duration'] / 60,
        best_game['sleep_score'],
        best_game['readiness_score'],
        best_game['pain'],
        best_game['soreness']
    ]
    
    worst_values = [
        worst_game['total_sleep_duration'] / 3600,
        worst_game['rem_sleep_duration'] / 60,
        worst_game['deep_sleep_duration'] / 60,
        worst_game['sleep_score'],
        worst_game['readiness_score'],
        worst_game['pain'],
        worst_game['soreness']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=f'Best Game ({best_game["date"].strftime("%m/%d")})',
        x=categories,
        y=best_values,
        marker=dict(color='#27ae60', line=dict(color='#000', width=1)),
        text=[f'{v:.1f}' for v in best_values],
        textposition='outside',
        textfont=dict(size=12, color='#000')
    ))
    
    fig.add_trace(go.Bar(
        name=f'Worst Game ({worst_game["date"].strftime("%m/%d")})',
        x=categories,
        y=worst_values,
        marker=dict(color='#e74c3c', line=dict(color='#000', width=1)),
        text=[f'{v:.1f}' for v in worst_values],
        textposition='outside',
        textfont=dict(size=12, color='#000')
    ))
    
    fig.update_layout(
        title=dict(text="Best vs Worst Game: Sleep & Recovery Comparison", font=dict(size=20, color='#000', family='Arial Black')),
        xaxis=dict(
            tickfont=dict(size=11, color='#000'),
            tickangle=-45
        ),
        yaxis=dict(
            title=dict(text="Value", font=dict(size=14, color='#000', family='Arial')),
            gridcolor='#e8e8e8',
            showgrid=True,
            tickfont=dict(size=12, color='#000')
        ),
        barmode='group',
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#000'),
        legend=dict(font=dict(size=12, color='#000'))
    )
    
    return fig, best_game, worst_game


def plot_correlation_heatmap(merged_df):
    """Correlation heatmap between sleep, pain, and performance metrics"""
    # Select relevant columns
    cols = ['total_sleep_duration', 'rem_sleep_duration', 'deep_sleep_duration', 
            'sleep_score', 'readiness_score', 'pain', 'soreness',
            'pts', 'total_rebs', 'ast', 'fg_pct', 'threept_pct', 'plus_minus']
    
    # Calculate correlation matrix
    corr_matrix = merged_df[cols].corr()
    
    # Create labels
    labels = ['Total Sleep', 'REM Sleep', 'Deep Sleep', 'Sleep Score', 'Readiness',
              'Pain', 'Soreness', 'Points', 'Rebounds', 'Assists', 'FG%', '3PT%', '+/-']
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont=dict(size=10, color='#000'),
        colorbar=dict(title='Correlation', tickfont=dict(size=12, color='#000'))
    ))
    
    fig.update_layout(
        title=dict(text="Correlation Matrix: Sleep, Pain & Performance", font=dict(size=20, color='#000', family='Arial Black')),
        height=600,
        width=700,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10, color='#000')),
        yaxis=dict(tickfont=dict(size=10, color='#000')),
        font=dict(color='#000')
    )
    
    return fig


def generate_insights(merged_df):
    """Generate data-driven narrative insights"""
    insights = []
    
    # Sleep vs Performance
    sleep_pts_corr = merged_df[['total_sleep_duration', 'pts']].corr().iloc[0, 1]
    if abs(sleep_pts_corr) > 0.3:
        direction = "positive" if sleep_pts_corr > 0 else "negative"
        insights.append(f"**Sleep-Performance Link**: There's a {direction} correlation ({sleep_pts_corr:.2f}) between total sleep and points scored.")
    
    # Pain impact
    pain_fg_corr = merged_df[['pain', 'fg_pct']].corr().iloc[0, 1]
    if abs(pain_fg_corr) > 0.3:
        direction = "negatively" if pain_fg_corr < 0 else "positively"
        insights.append(f"**Pain Impact**: Pain levels are {direction} correlated ({pain_fg_corr:.2f}) with shooting efficiency.")
    
    # Best game analysis
    best_idx = merged_df['pts'].idxmax()
    best_sleep = merged_df.loc[best_idx, 'total_sleep_duration'] / 3600
    avg_sleep = merged_df['total_sleep_duration'].mean() / 3600
    if best_sleep > avg_sleep:
        insights.append(f"**Best Performance**: Your highest-scoring game came after {best_sleep:.1f}h of sleep, which is {best_sleep - avg_sleep:.1f}h above your average.")
    
    # Recovery patterns
    avg_soreness = merged_df['soreness'].mean()
    high_soreness_games = merged_df[merged_df['soreness'] > avg_soreness]
    if len(high_soreness_games) > 0:
        avg_performance_sore = high_soreness_games['pts'].mean()
        avg_performance_normal = merged_df[merged_df['soreness'] <= avg_soreness]['pts'].mean()
        diff = avg_performance_normal - avg_performance_sore
        insights.append(f"**Recovery Effect**: You score {abs(diff):.1f} {'fewer' if diff > 0 else 'more'} points on average when soreness is above normal.")
    
    return insights


def main():
    st.title("Sleep, Pain & Performance Analytics")
    st.markdown("**Explorative Case Study**: How sleep quality and recovery impact basketball performance")

    #Background info
    st.markdown("""
                    This dashboard analyzes the relationships between sleep metrics (total sleep, REM, deep sleep), pain and soreness levels, and basketball performance metrics. 
                    I am currently dealing with chronic pain in my left wrist, and I want to understand how sleep affects the neurological pain signals and also my on-court performance. 
                    Last year, I struggled with poor sleep and inconsistent performance. So this year, I am tracking my sleep with my Oura Ring and documenting how my season goes.
                    """)
    
        #Background info
    st.markdown("""
                    This will be an ongoing project throughout my season. It will be fully automated using the Oura API and 
                    scraping data from gettysburgathletics.com for my game logs. More to come --- stay tuned!
                    """)
    
    # Try to load game data
    try:
        game_df = load_game_data()
        # Data loaded successfully - no need to display message
    except FileNotFoundError:
        st.error("❌ Could not find `data/game_data_log.csv`. Please add your game data.")
        st.info("Expected CSV columns: date, team, venue, minutes, pts, orebs, drebs, stl, ast, blk, tov, pf, fg_attempted, fgm, threept_attempted, threeptm, ftm, ft_attempted, plus_minus, total_sleep, rem_sleep, deep_sleep, light_sleep, awake_time, sleep_score, readiness_score, efficiency, pain, soreness")
        st.stop()
    except Exception as e:
        st.error(f"Error loading game data: {e}")
        st.stop()
    
    # Use CSV data only (no API fetching)
    merged_df = game_df
    
    # Show data summary
    if len(merged_df) == 0:
        st.error("No games found in the CSV file.")
        st.stop()
    
    st.markdown("---")
    st.subheader(f"Analysis Summary: {len(merged_df)} Games")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Points", f"{merged_df['pts'].mean():.1f}")
    with col2:
        st.metric("Avg Sleep", f"{merged_df['total_sleep_duration'].mean() / 3600:.1f}h")
    with col3:
        st.metric("Avg FG%", f"{merged_df['fg_pct'].mean():.1f}%")
    with col4:
        st.metric("Avg Pain", f"{merged_df['pain'].mean():.1f}")
    
    # Show insights first
    st.markdown("---")
    st.subheader("🧠 Data-Driven Insights")
    
    if len(merged_df) >= 3:
        insights = generate_insights(merged_df)
        if insights:
            for insight in insights:
                st.markdown(f"- {insight}")
        else:
            st.info("Not enough variation in data to generate meaningful insights yet. Keep logging games!")
    else:
        st.info("Will begin at 3 games logged.")
    
    # Visualizations
    st.markdown("---")
    
    # Sleep vs Performance
    st.markdown("#### 1. Sleep Duration vs Points Scored")
    st.plotly_chart(plot_sleep_vs_performance(merged_df), use_container_width=True)
    
    # Shooting Efficiency Matrix
    st.markdown("#### 2. Shooting Efficiency vs Sleep & Recovery")
    st.caption("Bubble size = soreness level, color = pain level (red = high, green = low)")
    st.plotly_chart(plot_shooting_efficiency_matrix(merged_df), use_container_width=True)
    
    # Best vs Worst
    if len(merged_df) >= 2:
        st.markdown("#### 3. Best vs Worst Game Comparison")
        fig_compare, best_game, worst_game = plot_best_worst_comparison(merged_df)
        st.plotly_chart(fig_compare, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Best Game**: {best_game['date'].strftime('%m/%d')} vs {best_game['team']}")
            st.write(f"- {best_game['pts']:.0f} pts, {best_game['fg_pct']:.1f}% FG")
            st.write(f"- {best_game['total_sleep_duration'] / 3600:.1f}h sleep")
        with col2:
            st.error(f"**Worst Game**: {worst_game['date'].strftime('%m/%d')} vs {worst_game['team']}")
            st.write(f"- {worst_game['pts']:.0f} pts, {worst_game['fg_pct']:.1f}% FG")
            st.write(f"- {worst_game['total_sleep_duration'] / 3600:.1f}h sleep")
    
    # Correlation Heatmap
    if len(merged_df) >= 3:
        st.markdown("#### 4. Full Correlation Matrix")
        st.caption("Shows relationships between all sleep, recovery, and performance variables")
        st.plotly_chart(plot_correlation_heatmap(merged_df), use_container_width=True)
    
    # Raw data table
    st.markdown("---")
    st.subheader("Game Log")
    display_cols = ['date', 'team', 'venue', 'pts', 'total_rebs', 'ast', 'fg_pct', 'threept_pct', 
                    'total_sleep_duration', 'sleep_score', 'pain', 'soreness', 'plus_minus']
     
    # Format for display
    display_df = merged_df[display_cols].copy()
    display_df['date'] = display_df['date'].dt.strftime('%m/%d/%Y')
    display_df['total_sleep_duration'] = (display_df['total_sleep_duration'] / 3600).round(1)
    display_df = display_df.rename(columns={
        'total_sleep_duration': 'sleep_hours',
        'total_rebs': 'reb'
    })
    
    st.dataframe(display_df, use_container_width=True)


if __name__ == "__main__":
    main()
