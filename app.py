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
    page_title="Traffic GNN | Nexus",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Premium CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Outfit:wght@500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #E2E8F0;
    }
    
    .stApp {
        background: radial-gradient(circle at top, #1E1E2E, #0F0F1A);
    }
    
    /* Header Typography */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Metric Cards - Glassmorphism */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: rgba(78, 205, 196, 0.5);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 15, 26, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(255, 107, 107, 0.5);
    }
    
    /* Clean up the main padding */
    .main .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- HEADER -----------------
col_title, col_status = st.columns([4, 1])
with col_title:
    st.title("Nexus AI: Spatio-Temporal Traffic Forecaster")
    st.markdown("<p style='color: #94A3B8; font-size: 1.1rem;'>Real-time Deep Graph Neural Network predictions on the METR-LA dataset</p>", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("## 🎛️ Control Center")
    st.markdown("Select parameters to visualize the live data stream and predictions.")
    
    sensor_id = st.number_input("📡 Target Sensor ID (0-206)", min_value=0, max_value=206, value=100)
    enable_auto_refresh = st.checkbox("🔄 Enable Live Updates", value=True)
    
    st.markdown("---")
    st.markdown("### 🛠️ System Architecture")
    st.markdown("""
    - **Model:** ST-GCN
    - **Spatial nodes:** 207 Sensors
    - **History:** 60 Mins (12 steps)
    - **Forecast:** 30 Mins (6 steps)
    """)
    st.markdown("---")
    if not enable_auto_refresh:
        refresh_btn = st.button("Manual Refresh ⚡")
    else:
        refresh_btn = False

# ----------------- DATA FETCHING -----------------
@st.cache_data(ttl=1)
def fetch_data():
    try:
        response = requests.get(API_URL, timeout=2)
        if response.status_code == 200:
            return response.json(), "🟢 Backend Online"
    except requests.exceptions.ConnectionError:
        return None, "🔴 Backend Offline"
    except Exception as e:
        return None, f"🟡 Backend Error: {str(e)}"
    return None, "🔴 Backend Offline"

# Simulate data if backend is offline to keep the UI beautiful
def generate_mock_data(time_step=100, sensor=100):
    np.random.seed(int(time.time() * 100) % (2**32))
    base_speed = 60 + np.sin(time_step/10.0) * 15
    history = [base_speed + np.random.normal(0, 3) for _ in range(12)]
    forecast = [history[-1] - i*0.5 + np.random.normal(0, 1) for i in range(1, 7)]
    return history, forecast, time_step

def build_dashboard():
    data, status_msg = fetch_data()
    
    with col_status:
        st.markdown(f"<div style='text-align: right; margin-top: 1rem; font-weight: 600;'>{status_msg}</div>", unsafe_allow_html=True)
    
    if data is None:
        st.warning("⚠️ Cannot connect to backend. Showing simulation mode data for demonstration.")
        past_values, pred_values, time_step = generate_mock_data(int(time.time()), sensor_id)
    else:
        input_window = data.get("input_window", [])
        prediction = data.get("prediction", [])
        time_step = data.get("time_step", 0)

        if not input_window:
            st.info("⏳ Waiting for data stream... Start the producer.py script.")
            return

        past_values = [step[sensor_id] for step in input_window]
        pred_values = [step[sensor_id] for step in prediction] if prediction else []

    current_time_idx = time_step
    past_indices = list(range(current_time_idx - len(past_values) + 1, current_time_idx + 1))
    future_indices = list(range(current_time_idx + 1, current_time_idx + 1 + len(pred_values)))

    # ----------------- KEY METRICS -----------------
    current_speed = past_values[-1]
    avg_hist_speed = sum(past_values)/len(past_values)
    avg_pred_speed = sum(pred_values)/len(pred_values) if pred_values else 0
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Speed", f"{current_speed:.1f} mph", f"{current_speed - past_values[-2]:.1f} mph" if len(past_values)>1 else "")
    m2.metric("Avg Forecast (30m)", f"{avg_pred_speed:.1f} mph", f"{avg_pred_speed - current_speed:.1f} mph" if pred_values else "")
    m3.metric("Traffic Trend", "Congesting" if avg_pred_speed < current_speed else "Clearing", delta_color="inverse" if avg_pred_speed < current_speed else "normal")
    m4.metric("Time Step", f"{time_step}", "Live")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ----------------- MAIN CHARTS -----------------
    # Create Plotly Chart
    fig = go.Figure()

    # Historical Line
    fig.add_trace(go.Scatter(
        x=past_indices, 
        y=past_values,
        mode='lines+markers',
        name='Actual History (60m)',
        line=dict(color='#4ECDC4', width=4, shape='spline'),
        marker=dict(size=8, color='#4ECDC4', line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor='rgba(78, 205, 196, 0.1)'
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
            line=dict(color='#FF6B6B', width=4, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=future_indices,
            y=pred_values,
            mode='lines+markers',
            name='GNN Forecast (30m)',
            line=dict(color='#FF6B6B', width=4, dash='dot', shape='spline'),
            marker=dict(size=10, symbol='diamond', color='#FF6B6B', line=dict(width=2, color='white'))
        ))

    fig.update_layout(
        title=dict(text=f"Traffic Speed Trajectory - Sensor #{sensor_id}", font=dict(family="Outfit", size=24, color='white')),
        xaxis_title="Time Step (5-min intervals)",
        yaxis_title="Speed (mph)",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=550,
        hovermode="x unified",
        legend=dict(
            yanchor="top", y=0.99, 
            xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # ----------------- DATA TABLE -----------------
    st.markdown("### 📊 Detailed Forecast Data")
    if pred_values:
        df_pred = pd.DataFrame({
            "Time Step": future_indices,
            "Minutes from Now": [(i+1)*5 for i in range(len(future_indices))],
            "Predicted Speed (mph)": [round(val, 2) for val in pred_values]
        })
        st.dataframe(df_pred.style.highlight_min(subset=["Predicted Speed (mph)"], color="#FF6B6B")
                                     .highlight_max(subset=["Predicted Speed (mph)"], color="#4ECDC4"), 
                     use_container_width=True, hide_index=True)

if __name__ == "__main__":
    build_dashboard()
    
    if enable_auto_refresh:
        time.sleep(REFRESH_RATE)
        st.rerun()
