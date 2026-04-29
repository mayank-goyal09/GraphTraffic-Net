import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from models.st_gcn import TrafficForecaster
from utils.data_utils import load_dataset, get_graph_structure

# 1. Configuration & Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32 # Reduced for stability
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 10 # Early stopping patience

NUM_NODES = 207
INPUT_TIMESTEPS = 12  # 60 mins history
OUTPUT_TIMESTEPS = 6  # 30 mins forecast

def train():
    # 2. Load Real/Synthetic Data
    data, scaler = load_dataset('data/', BATCH_SIZE)
    
    # Load Graph Structure
    edge_index, edge_weight = get_graph_structure('data/')
    edge_index = edge_index.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)
    
    # 3. Initialize Model, Optimizer, and Loss
    # Feature dimension is 1 for our current synthetic data
    model = TrafficForecaster(NUM_NODES, 1, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Adjust the Loss Function (Switch to Huber Loss) ⚖️
    criterion = nn.HuberLoss() # Robust to outliers

    print(f"🚀 Training started on {DEVICE}...")
    
    # Early Stopping Variables
    best_val_loss = float('inf')
    patience_counter = 0
    model_path = "models/traffic_model.pth"

    for epoch in range(EPOCHS):
        model.train()
        train_loss = []
        
        # Iterate through batches
        for i in range(0, len(data['x_train']), BATCH_SIZE):
            x_batch = torch.FloatTensor(data['x_train'][i:i+BATCH_SIZE]).to(DEVICE)
            y_batch = torch.FloatTensor(data['y_train'][i:i+BATCH_SIZE]).to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward Pass
            output = model(x_batch, edge_index, edge_weight)
            loss = criterion(output, y_batch)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())

        # 4. Validation Phase
        model.eval()
        val_loss = []
        with torch.no_grad():
            for i in range(0, len(data['x_val']), BATCH_SIZE):
                x_val = torch.FloatTensor(data['x_val'][i:i+BATCH_SIZE]).to(DEVICE)
                y_val = torch.FloatTensor(data['y_val'][i:i+BATCH_SIZE]).to(DEVICE)
                output = model(x_val, edge_index, edge_weight)
                loss = criterion(output, y_val)
                val_loss.append(loss.item())
        
        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)

        print(f"✅ Epoch {epoch+1:02d}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")
        
        # 2. Implement "Early Stopping" ⏱️
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"   💾 New Best Model Saved! (Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement. Patience: {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print("🛑 Early stopping triggered.")
            break

    print(f"🏆 Training Complete. Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
