#!/bin/bash

uvicorn app.main:app --host 0.0.0.0 --port 8000 &

echo "🕒 Waiting for backend to become ready..."
while ! nc -z localhost 8000; do
  sleep 1
done

echo "✅ Backend is up. Starting frontend..."
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

echo "✅ Application is up. Go to http://localhost:8501/"
wait
