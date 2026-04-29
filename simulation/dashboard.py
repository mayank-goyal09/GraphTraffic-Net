import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# Configuration
API_URL = "http://localhost:8000/predict"
REFRESH_RATE = 1.0 # Seconds

st.set_page_config(
    page_title="Real-Time Traffic Forecaster",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stApp {
        background-color: #0E1117;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚦 Real-Time Traffic GNN Forecaster")
st.markdown("### 📡 Spatio-Temporal Graph Neural Network Simulation")

# Sidebar
st.sidebar.header("Control Panel")
sensor_id = st.sidebar.number_input("Select Sensor ID to Monitor (0-206)", min_value=0, max_value=206, value=100)
enable_auto_refresh = st.sidebar.checkbox("Enable Live Updates", value=True)

# Main columns
col1, col2 = st.columns([3, 1])

with col1:
    chart_placeholder = st.empty()

with col2:
    stats_placeholder = st.empty()

def fetch_data():
    try:
        response = requests.get(API_URL)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return None
    return None

def update_dashboard():
    data = fetch_data()
    
    if data is None:
        chart_placeholder.warning("⚠️ Cannot connect to Backend. Is it running?")
        return

    input_window = data.get("input_window", [])
    prediction = data.get("prediction", [])
    time_step = data.get("time_step", 0)

    if not input_window:
        chart_placeholder.info("⏳ Waiting for data stream...")
        return

    # Process data for the selected sensor
    # Input window: [12, 207]
    past_values = [step[sensor_id] for step in input_window]
    
    # Prediction: [6, 207]
    pred_values = [step[sensor_id] for step in prediction] if prediction else []
    
    # Time axis
    # Assume 5-min intervals. 
    # Past: t-11 to t
    # Future: t+1 to t+6
    
    current_time_idx = time_step
    past_indices = list(range(current_time_idx - len(past_values) + 1, current_time_idx + 1))
    future_indices = list(range(current_time_idx + 1, current_time_idx + 1 + len(pred_values)))
    
    # Create Plotly Chart
    fig = go.Figure()

    # Historical Line
    fig.add_trace(go.Scatter(
        x=past_indices, 
        y=past_values,
        mode='lines+markers',
        name='Actual History',
        line=dict(color='#00CC96', width=3),
        marker=dict(size=6)
    ))

    # Prediction Line
    if pred_values:
        # Connect last actual to first prediction
        connector_x = [past_indices[-1], future_indices[0]]
        connector_y = [past_values[-1], pred_values[0]]
        
        fig.add_trace(go.Scatter(
            x=connector_x,
            y=connector_y,
            mode='lines',
            showlegend=False,
            line=dict(color='#AB63FA', width=3, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=future_indices,
            y=pred_values,
            mode='lines+markers',
            name='Forecast (30 min)',
            line=dict(color='#AB63FA', width=3, dash='dot'),
            marker=dict(size=6, symbol='diamond')
        ))

    fig.update_layout(
        title=f"Sensor #{sensor_id} - Traffic Speed",
        xaxis_title="Time Step (5-min intervals)",
        yaxis_title="Speed (mph)",
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    chart_placeholder.plotly_chart(fig, use_container_width=True)

    # Stats Panel
    with stats_placeholder.container():
        st.markdown(f"### ⏱️ Time Step: {time_step}")
        if past_values:
            st.metric("Current Speed", f"{past_values[-1]:.2f} mph")
        
        if pred_values:
            avg_pred = sum(pred_values)/len(pred_values)
            st.metric("Avg Forecast Speed", f"{avg_pred:.2f} mph", 
                      delta=f"{avg_pred - past_values[-1]:.2f}" if past_values else None)
            
            st.markdown("#### 🔮 Forecast Steps")
            for i, val in enumerate(pred_values):
                st.text(f"+{(i+1)*5} min: {val:.1f} mph")

if __name__ == "__main__":
    if enable_auto_refresh:
        update_dashboard()
        time.sleep(REFRESH_RATE)
        st.rerun()
    else:
        if st.button("Refresh Now"):
            update_dashboard()
