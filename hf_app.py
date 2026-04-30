import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import h5py
import os
import time
from models.st_gcn import TrafficForecaster
from utils.data_utils import get_graph_structure, download_from_drive

# --- CONFIGURATION ---
NUM_NODES = 207
INPUT_TIMESTEPS = 12
OUTPUT_TIMESTEPS = 6
NUM_FEATURES = 1
MODEL_PATH = "models/traffic_model.pth"
DATA_DIR = "data"
H5_PATH = os.path.join(DATA_DIR, 'metr-la.h5')

st.set_page_config(
    page_title="Nexus AI | Hugging Face",
    page_icon="🚦",
    layout="wide"
)

# --- PREMIUM CSS ---
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
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    device = torch.device("cpu")
    
    # 1. Ensure Model Exists
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model weights from Google Drive...")
        download_from_drive("1K0JwT7E6sO2jb7rwyuGj1wmw4l6hBv8F", MODEL_PATH)
    
    # 2. Load Model
    model = TrafficForecaster(NUM_NODES, NUM_FEATURES, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 3. Load Graph
    edge_index, edge_weight = get_graph_structure(DATA_DIR)
    
    # 4. Load Data Stats
    with h5py.File(H5_PATH, 'r') as f:
        full_data = f['df'][:]
        mean = float(full_data.mean())
        std = float(full_data.std())
        
    return model, edge_index, edge_weight, mean, std, full_data

# --- INFERENCE ENGINE ---
def run_inference(model, edge_index, edge_weight, mean, std, input_window):
    # input_window: [12, 207]
    norm_input = (input_window - mean) / std
    x_tensor = torch.FloatTensor(norm_input).unsqueeze(0).unsqueeze(-1) # [1, 12, 207, 1]
    
    with torch.no_grad():
        pred_tensor = model(x_tensor, edge_index, edge_weight)
        
    pred_array = pred_tensor.numpy().squeeze(0) # [6, 207]
    denorm_pred = (pred_array * std) + mean
    return denorm_pred

# --- MAIN UI ---
def main():
    st.title("Nexus AI: ST-GCN Traffic Forecasting")
    st.markdown("### 🏙️ Hugging Face Deployment Edition")
    
    try:
        model, edge_index, edge_weight, mean, std, full_data = load_resources()
    except Exception as e:
        st.error(f"❌ Error loading resources: {e}")
        return

    # Sidebar Controls
    with st.sidebar:
        st.markdown("## 🎮 Simulation Hub")
        sensor_id = st.number_input("📡 Target Sensor ID", 0, NUM_NODES-1, 100)
        
        if st.button("🚀 Sample Random Traffic Event"):
            # Pick a random window
            max_idx = len(full_data) - INPUT_TIMESTEPS - OUTPUT_TIMESTEPS
            start_idx = np.random.randint(0, max_idx)
            st.session_state.start_idx = start_idx
        
        if 'start_idx' not in st.session_state:
            st.session_state.start_idx = 1000 # Default

    # Extract Window
    s_idx = st.session_state.start_idx
    input_window = full_data[s_idx : s_idx + INPUT_TIMESTEPS] # [12, 207]
    ground_truth = full_data[s_idx + INPUT_TIMESTEPS : s_idx + INPUT_TIMESTEPS + OUTPUT_TIMESTEPS] # [6, 207]
    
    # Run Model
    prediction = run_inference(model, edge_index, edge_weight, mean, std, input_window)

    # UI Metrics
    m1, m2, m3 = st.columns(3)
    curr_speed = input_window[-1, sensor_id]
    pred_avg = prediction[:, sensor_id].mean()
    
    m1.metric("Current Speed", f"{curr_speed:.1f} mph")
    m2.metric("30-min Forecast", f"{pred_avg:.1f} mph", f"{pred_avg - curr_speed:.1f} mph")
    m3.metric("Dataset Index", f"T-{s_idx}")

    # Plotly Visuals
    past_indices = list(range(-11, 1))
    future_indices = list(range(1, 7))
    
    fig = go.Figure()
    
    # History
    fig.add_trace(go.Scatter(
        x=past_indices, y=input_window[:, sensor_id],
        mode='lines+markers', name='History (60m)',
        line=dict(color='#4ECDC4', width=4, shape='spline'),
        fill='tozeroy', fillcolor='rgba(78, 205, 196, 0.1)'
    ))
    
    # Ground Truth
    fig.add_trace(go.Scatter(
        x=future_indices, y=ground_truth[:, sensor_id],
        mode='lines+markers', name='Actual Future',
        line=dict(color='#E2E8F0', width=2, dash='dash')
    ))
    
    # Prediction
    fig.add_trace(go.Scatter(
        x=future_indices, y=prediction[:, sensor_id],
        mode='lines+markers', name='GNN Forecast',
        line=dict(color='#FF6B6B', width=4, dash='dot')
    ))

    fig.update_layout(
        title=f"Spatio-Temporal Analysis: Sensor #{sensor_id}",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Relative Time (5-min steps)",
        yaxis_title="Speed (mph)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    #### 💡 How it works on Hugging Face:
    This space uses a **Spatio-Temporal Graph Convolutional Network (ST-GCN)**. Unlike standard LSTMs, this model looks at the **Graph Adjacency Matrix** of the 207 sensors to understand how traffic flows from one road to another.
    """)

if __name__ == "__main__":
    main()
