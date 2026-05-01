import sys
import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from collections import deque
from typing import List
import uvicorn
import h5py

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.append(PROJECT_ROOT)

from models.st_gcn import TrafficForecaster
from utils.data_utils import get_graph_structure

app = FastAPI(title="Traffic Prediction API", description="Real-time ST-GCN Inference Backend")

# --- Global State ---
DEVICE = torch.device("cpu") # Use CPU for inference for simplicity/compatibility
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'traffic_model.pth')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Constants from training
NUM_NODES = 207
INPUT_TIMESTEPS = 12
OUTPUT_TIMESTEPS = 6
NUM_FEATURES = 1 # Speed only

# Data Buffers
# Stores raw (unscaled) history for plotting "Actual"
history_buffer = deque(maxlen=INPUT_TIMESTEPS) 
# Stores latest prediction
latest_prediction = None
last_updated_step = -1

# Normalization stats (Loaded on startup)
global_mean = 0.0
global_std = 1.0
edge_index = None
edge_weight = None

# Model
model = None

# --- Schemas ---
class TrafficData(BaseModel):
    time_step: int
    data: List[float] # 207 sensor values

class PredictionResponse(BaseModel):
    time_step: int
    input_window: List[List[float]] # [12, 207]
    prediction: List[List[float]] # [6, 207]
    sensor_ids: List[int] # Indices of sensors

# --- Startup ---
@app.on_event("startup")
async def startup_event():
    global model, global_mean, global_std, edge_index, edge_weight
    
    # 1. Load Data Stats for Normalization
    try:
        h5_path = os.path.join(DATA_DIR, 'metr-la.h5')
        with h5py.File(h5_path, 'r') as f:
            data = f['df'][:]
            global_mean = float(data.mean())
            global_std = float(data.std())
        print(f"[INFO] Loaded Data Stats: Mean={global_mean:.4f}, Std={global_std:.4f}")
    except Exception as e:
        print(f"[WARN] Failed to load data stats: {e}. Using default.")

    # 2. Load Graph Structure
    try:
        edge_index, edge_weight = get_graph_structure(DATA_DIR)
        edge_index = edge_index.to(DEVICE)
        edge_weight = edge_weight.to(DEVICE)
        print("[INFO] Graph structure loaded.")
    except Exception as e:
        print(f"[ERROR] Failed to load graph structure: {e}")
        raise e

    # 3. Load Model
    try:
        model = TrafficForecaster(NUM_NODES, NUM_FEATURES, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS)
        if not os.path.exists(MODEL_PATH):
            print("[WARN] Model weights not found locally. Downloading from Google Drive...")
            from utils.data_utils import download_from_drive
            download_from_drive("1K0JwT7E6sO2jb7rwyuGj1wmw4l6hBv8F", MODEL_PATH)
            
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            print(f"[INFO] Model loaded from {MODEL_PATH}")
        else:
            print(f"[WARN] Model path {MODEL_PATH} not found or could not be downloaded. Running with initialized weights.")
            model.to(DEVICE)
            model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise e

# --- Endpoints ---

@app.post("/ingest")
async def ingest_data(payload: TrafficData):
    global latest_prediction, last_updated_step
    
    # 1. Update Buffer
    # We store the raw data row. 
    # Validating size
    if len(payload.data) != NUM_NODES:
        raise HTTPException(status_code=400, detail=f"Expected {NUM_NODES} sensor values, got {len(payload.data)}")
        
    history_buffer.append(payload.data)
    last_updated_step = payload.time_step
    
    # 2. Run Inference if we have enough data
    if len(history_buffer) == INPUT_TIMESTEPS:
        # Prepare input tensor: [Batch=1, Time=12, Nodes=207, Feat=1]
        input_data = np.array(history_buffer) # [12, 207]
        
        # Normalize
        norm_input = (input_data - global_mean) / global_std
        
        # Add dimensions
        # Shape: [12, 207] -> [1, 12, 207, 1]
        x_tensor = torch.FloatTensor(norm_input).unsqueeze(0).unsqueeze(-1).to(DEVICE)
        
        with torch.no_grad():
            # Output: [Batch=1, Output_Steps=6, Nodes=207]
            pred_tensor = model(x_tensor, edge_index, edge_weight)
            
        # Denormalize
        pred_array = pred_tensor.cpu().numpy().squeeze(0) # [6, 207]
        denorm_pred = (pred_array * global_std) + global_mean
        
        latest_prediction = denorm_pred.tolist()
        
    return {"status": "ok", "buffer_size": len(history_buffer)}

@app.get("/predict", response_model=PredictionResponse)
async def get_prediction():
    if latest_prediction is None:
        # Return empty/zeros if not ready
        return {
            "time_step": last_updated_step,
            "input_window": list(history_buffer),
            "prediction": [],
            "sensor_ids": list(range(NUM_NODES))
        }
    
    return {
        "time_step": last_updated_step,
        "input_window": list(history_buffer),
        "prediction": latest_prediction,
        "sensor_ids": list(range(NUM_NODES))
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
