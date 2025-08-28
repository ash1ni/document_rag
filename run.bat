@echo off
echo Starting Chat with PDF - RAG System...
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Checking dependencies...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo Starting Qdrant vector database...
echo If you haven't started Qdrant yet, please run:
echo docker-compose up -d
echo.
echo Or start Qdrant manually and ensure it's running on localhost:6333
echo.

echo Starting Streamlit application...
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run app.py

pause 