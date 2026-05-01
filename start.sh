#!/bin/bash

# Ensure we are in the app directory
cd /app

# 1. Start the FastAPI Backend in the background
echo "Starting FastAPI Backend..."
export PYTHONUTF8=1
python simulation/backend.py > backend.log 2>&1 &

# 2. Start the Mock Data Producer in the background
echo "Starting Data Producer..."
python simulation/producer.py > producer.log 2>&1 &

# 3. Start the Streamlit Dashboard
echo "Starting Streamlit Dashboard..."
# Hugging Face Spaces uses port 7860
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
