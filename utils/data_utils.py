import os
import numpy as np
import pandas as pd
import pickle
import h5py
import torch

def generate_synthetic_data(data_dir='data'):
    """Generates synthetic METR-LA style data for testing purposes."""
    os.makedirs(os.path.join(data_dir, 'sensor_graph'), exist_ok=True)
    
    # 1. Generate adj_mx.pkl (207 sensors)
    adj_path = os.path.join(data_dir, 'sensor_graph/adj_mx.pkl')
    if not os.path.exists(adj_path):
        print("Generating SPARSE synthetic adj_mx.pkl...")
        sensor_ids = [str(i) for i in range(207)]
        sensor_id_to_ind = {id: i for i, id in enumerate(sensor_ids)}
        
        # Create a sparse adjacency matrix (road network style)
        adj_mx = np.zeros((207, 207))
        for i in range(207):
            adj_mx[i, i] = 1.0 # Self-loop
            # Connect to a few nearby neighbors
            for j in range(max(0, i-5), min(207, i+6)):
                if np.random.random() > 0.3:
                    weight = np.random.uniform(0.1, 0.9)
                    adj_mx[i, j] = weight
                    adj_mx[j, i] = weight
        
        with open(adj_path, 'wb') as f:
            pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f)
        print(f"Generated SPARSE {adj_path}")


    # 2. Generate metr-la.h5 (Mock traffic speeds)
    h5_path = os.path.join(data_dir, 'metr-la.h5')
    if not os.path.exists(h5_path):
        print("Generating synthetic metr-la.h5...")
        # METR-LA usually has ~34272 time steps (4 months @ 5min)
        # We'll generate a smaller sample for testing: 1000 steps
        num_steps = 2000 
        data = np.random.normal(50, 15, (num_steps, 207))
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('df', data=data)
        print(f"Generated {h5_path}")

def get_graph_structure(data_dir='data'):
    """Loads adj_mx and converts to edge_index and edge_weight."""
    adj_path = os.path.join(data_dir, 'sensor_graph/adj_mx.pkl')
    with open(adj_path, 'rb') as f:
        _, _, adj_mx = pickle.load(f, encoding='latin1')
    
    # Convert to COO format for PyTorch Geometric
    from scipy.sparse import coo_matrix
    coo = coo_matrix(adj_mx)
    edge_index = torch.LongTensor(np.vstack((coo.row, coo.col)))
    edge_weight = torch.FloatTensor(coo.data)
    return edge_index, edge_weight

def load_dataset(data_dir, batch_size, input_steps=12, output_steps=6):
    """Loads and preprocesses the traffic dataset."""
    h5_path = os.path.join(data_dir, 'metr-la.h5')
    with h5py.File(h5_path, 'r') as f:
        data = f['df'][:]
    
    # Simple Z-Score Normalization
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std
    
    # Create windows
    x, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        x.append(data[i : i + input_steps])
        # We predict the next output_steps. METR-LA predicts speed for each sensor.
        # Shape of y: [Steps, Nodes]
        y.append(data[i + input_steps : i + input_steps + output_steps])
    
    x = np.array(x) # [Samples, Input_Steps, Nodes]
    y = np.array(y) # [Samples, Output_Steps, Nodes]
    
    # Reshape x to [Samples, Time, Nodes, Features] - user's model expects channels
    # But wait, METR-LA has speed and optionally time-of-day. 
    # Let's assume 1 feature (speed) for now to match synthetic data.
    x = np.expand_dims(x, axis=-1) 
    
    # Split
    split = int(0.7 * len(x))
    data_dict = {
        'x_train': x[:split],
        'y_train': y[:split],
        'x_val': x[split:],
        'y_val': y[split:]
    }
    
    return data_dict, (mean, std)

def download_info():
    print("\n--- Link to Real METR-LA Dataset ---")
    print("If you need the real research data, please download them manually from Mendeley Data:")
    print("1. adj_mx.pkl: https://data.mendeley.com/datasets/79m866pxxt/1")
    print("2. metr-la.h5: https://www.kaggle.com/datasets/oscarleo/metr-la")
    print("Place them in the 'data/' folder of this project.\n")

def download_from_drive(file_id, destination):
    """Downloads a large file from Google Drive using its file ID."""
    import requests
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
            
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
        
    CHUNK_SIZE = 32768
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
    print(f"Downloaded file to {destination}")

if __name__ == "__main__":
    generate_synthetic_data()
    download_info()


