# 🚦 Traffic-GNN-Forecaster 🚦

This project leverages a Spatio-Temporal Graph Convolutional Network (ST-GCN) to forecast real-time traffic speeds using deep learning. It operates on the standard **METR-LA** dataset.

## 🧠 Model Weights

Due to GitHub's 100MB limit on file sizes, the trained model weights (`traffic_model.pth`) are hosted externally. 

📥 **[Download Model Weights (traffic_model.pth) Here](https://drive.google.com/file/d/1K0JwT7E6sO2jb7rwyuGj1wmw4l6hBv8F/view?usp=sharing)**

*Place the downloaded file into the `models/` directory.*

---

## 📂 Project Structure

```
project/
├── data/               # METR-LA traffic data & sensor graph adjacency matrix
├── models/             # PyTorch ST-GCN model logic & saved weights
├── utils/              # Data pre-processing and scaling tools
├── simulation/         # 🚀 Live Streaming & Dashboard Pipeline
├── train.py            # Model training script (with Early Stopping)
├── predict.py          # Local inference testing script
└── requirements.txt    # Python environment dependencies
```

## 🚀 Quick Start

### 1. Install Requirements
Make sure you are in the root folder of the project and execute:
```bash
pip install -r requirements.txt
```

### 2. Make Predictions
Once you have placed `traffic_model.pth` inside `models/`, run a basic validation test:
```bash
python predict.py
```

### 3. Launch the Dashboard Simulation
Check out the complete setup guide inside the **[Simulation README](file:///c:/my_local_data%28one%20drive%29/Attachments/Ambition%20course/my_all_projects/project%2060%20Traffic-GNN-Forecaster/simulation/README.md)** to start up the FastAPI backend, streaming IoT node producers, and real-time Streamlit visualization dashboard.

---

## 🏗️ Training Your Own
To train a fresh model over the dataset from scratch:
```bash
python train.py
```
*Includes Huber Loss implementation for outlier stability.*
