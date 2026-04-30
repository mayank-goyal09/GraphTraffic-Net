#!/bin/bash

# 1. Start the FastAPI Backend in the background
echo "🚀 Starting FastAPI Backend..."
export PYTHONUTF8=1
python simulation/backend.py &

# 2. Start the Mock Data Producer in the background
echo "📡 Starting Data Producer..."
python simulation/producer.py &

# 3. Start the Streamlit Dashboard
echo "📊 Starting Streamlit Dashboard..."
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
