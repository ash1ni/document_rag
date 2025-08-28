#!/bin/bash

echo "Starting Chat with PDF - RAG System..."
echo

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "Python version: $(python3 --version)"
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check dependencies
echo "Checking dependencies..."
if ! pip show streamlit &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
fi

echo
echo "Starting Qdrant vector database..."
echo "If you haven't started Qdrant yet, please run:"
echo "docker-compose up -d"
echo
echo "Or start Qdrant manually and ensure it's running on localhost:6333"
echo

# Check if Qdrant is running
if ! curl -s http://localhost:6333/health &> /dev/null; then
    echo "WARNING: Qdrant doesn't seem to be running on localhost:6333"
    echo "Please start Qdrant before continuing"
    echo
    read -p "Press Enter to continue anyway..."
fi

echo "Starting Streamlit application..."
echo "The app will open in your browser at http://localhost:8501"
echo
echo "Press Ctrl+C to stop the application"
echo

streamlit run app.py 