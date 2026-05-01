import sys
import os
import time
import requests
import numpy as np
import h5py
import pandas as pd

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.append(PROJECT_ROOT)

# Configuration
API_URL = "http://localhost:8000/ingest"
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'metr-la.h5')
# Simulation speed: 1 second per time step (for demo purposes)
SIMULATION_DELAY = 1.0 

def mock_producer():
    """
    Reads data from .h5 file row-by-row and sends it to the FastAPI backend.
    """
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found at {DATA_PATH}. Please generate it first.")
        return

    print(f"[INFO] Starting Mock Producer using {DATA_PATH}...")
    
    with h5py.File(DATA_PATH, 'r') as f:
        # Assuming the dataset key is 'df' based on data_utils.py
        data = f['df'][:]
        
    total_steps = len(data)
    print(f"[INFO] Total time steps available: {total_steps}")
    
    for i in range(total_steps):
        # Extract current time step data (207 sensors)
        current_data = data[i]
        
        # Prepare payload
        payload = {
            "time_step": i,
            "data": current_data.tolist() # Convert numpy array to list
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                print(f"[INFO] Sent time step {i}/{total_steps}")
            else:
                print(f"[WARN] Failed to send time step {i}: {response.text}")
        except requests.exceptions.ConnectionError:
            print("[ERROR] Connection error. Is the backend running?")
        except Exception as e:
            print(f"[ERROR] Error: {e}")
            
        time.sleep(SIMULATION_DELAY)

if __name__ == "__main__":
    mock_producer()
