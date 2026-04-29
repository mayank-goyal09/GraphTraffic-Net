import torch
import numpy as np
import os
from models.st_gcn import TrafficForecaster
from utils.data_utils import load_dataset, get_graph_structure, download_from_drive

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_NODES = 207
INPUT_TIMESTEPS = 12
OUTPUT_TIMESTEPS = 6

def predict():
    print("Loading model and data...")
    # Load Data (using the scaler from training is important!)
    data, (mean, std) = load_dataset('data/', batch_size=1)
    
    # Load Graph
    edge_index, edge_weight = get_graph_structure('data/')
    edge_index = edge_index.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)

    # Initialize Model
    model = TrafficForecaster(NUM_NODES, 1, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS).to(DEVICE)
    
    model_path = "models/traffic_model.pth"
    if not os.path.exists(model_path):
        print("Model weights not found locally. Downloading from Google Drive...")
        download_from_drive("1K0JwT7E6sO2jb7rwyuGj1wmw4l6hBv8F", model_path)
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Get a sample input from Validation Set
    sample_idx = 0
    x_input = torch.FloatTensor(data['x_val'][sample_idx:sample_idx+1]).to(DEVICE)
    y_true = data['y_val'][sample_idx:sample_idx+1] # Authentic future values

    # Run Prediction
    with torch.no_grad():
        y_pred = model(x_input, edge_index, edge_weight)
        y_pred = y_pred.cpu().numpy()

    # De-normalize (Back to real units e.g. mph)
    y_pred_denorm = (y_pred * std) + mean
    y_true_denorm = (y_true * std) + mean

    print(f"\nPrediction Result for Sensor #0 (First 30 mins):")
    print("-" * 50)
    print(f"{'Time (min)':<15} | {'Predicted Speed':<20} | {'Actual Speed':<20}")
    print("-" * 50)
    
    time_steps = []
    preds = []
    actuals = []
    
    for t in range(OUTPUT_TIMESTEPS):
        pred_val = y_pred_denorm[0, t, 0] # Batches, Time, Node 0
        true_val = y_true_denorm[0, t, 0]
        print(f"{(t+1)*5:<15} | {pred_val:.2f} mph{' '*12} | {true_val:.2f} mph")
        
        time_steps.append((t+1)*5)
        preds.append(pred_val)
        actuals.append(true_val)
    
    print("-" * 50)
    diff = np.mean(np.abs(y_pred_denorm - y_true_denorm))
    print(f"Average Error across all sensors: {diff:.2f} mph")
    
    # Plotting
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(time_steps, actuals, marker='o', label='Actual Traffic', color='red', linewidth=2)
        plt.plot(time_steps, preds, marker='s', label='GNN Prediction', color='blue', linestyle='--')
        plt.title(f'Traffic Speed Prediction (Sensor #0)\nAvg Error: {diff:.2f} mph')
        plt.xlabel('Time (Minutes)')
        plt.ylabel('Speed (MPH)')
        plt.legend()
        plt.grid(True)
        plt.savefig('prediction_plot.png')
        print("Plot saved to 'prediction_plot.png'")
    except ImportError:
        print("Matplotlib not installed, skipping plot.")

if __name__ == "__main__":
    predict()

