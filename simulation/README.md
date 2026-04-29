# 🚦 Real-Time Traffic Forecasting Simulation Pipeline

This project demonstrates a complete MLOps pipeline for real-time traffic speed prediction using a Spatio-Temporal Graph Convolutional Network (ST-GCN).

## 📂 Architecture

The simulation consists of three modular components:

1.  **Mock Producer (`producer.py`)**: 
    - Simulates IoT sensors by reading historical traffic data (`metr-la.h5`) row-by-row.
    - Streams data points to the backend API every second.

2.  **Inference Backend (`backend.py`)**: 
    - Built with **FastAPI**.
    - Maintains a sliding window state (last 12 time steps / 60 mins).
    - Runs inference using the trained `STGCN` model (`traffic_model.pth`).
    - Returns 30-minute forecasts (next 6 time steps).

3.  **Live Dashboard (`dashboard.py`)**: 
    - Built with **Streamlit**.
    - Polls the backend for the latest sensor readings and predictions.
    - visualize "Actual vs. Predicted" speeds for any of the 207 sensors.

---

## 🚀 Setup & Usage

### 1. Install Dependencies
Ensure you are in the project root directory.
```bash
pip install -r simulation/requirements.txt
```

### 2. Start the Backend Server
Open a terminal and run:
```bash
python simulation/backend.py
```
*The server will start at `http://localhost:8000`.*

### 3. Start the Mock Producer
Open a **second** terminal and run:
```bash
python simulation/producer.py
```
*This will start sending data to the backend.*

### 4. Launch the Dashboard
Open a **third** terminal and run:
```bash
streamlit run simulation/dashboard.py
```
*The dashboard will open in your browser.*

---

## 🧠 Model Details

### Spatio-Temporal GCN Logic
The core of this system is the **ST-GCN** (Spatio-Temporal Graph Convolutional Network), designed to handle the complex dependencies in traffic data:
- **Spatial Dependency**: Modeled using a graph structure where nodes are sensors and edges are road connections. We use **Graph Attention Networks (GAT)** or **ChebConv** to aggregate information from neighboring sensors.
- **Temporal Dependency**: Modeled using **1D Temporal Convolutions (TCN)** with Gated Linear Units (GLU) to capture trends over time.

### Training Strategy
- **Dataset**: METR-LA (207 sensors).
- **Validation Loss**: 0.55 (MAE/Huber).
- **Early Stopping**: We employed Early Stopping with a patience of 10 epochs. This prevented overfitting by halting training when the validation loss stopped improving, ensuring the model generalizes well to unseen data.

---

## 🛠️ File Structure

```
project/
├── data/               # Contains metr-la.h5 and sensor graph
├── models/             # PyTorch model definitions (st_gcn.py)
├── utils/              # Data utilities
├── simulation/         # 🚀 Simulation Pipeline (New)
│   ├── backend.py      # FastAPI Inference Server
│   ├── dashboard.py    # Streamlit Visualization
│   ├── producer.py     # Data Stream Simulation
│   └── requirements.txt
└── train.py            # Training script
```
