@echo off
REM ML Bootcamp - Quick Start Script for Windows
REM This script will set up and run the ML bootcamp in one command

echo ==========================================
echo   ML Bootcamp - Quick Start
echo   Setting up your learning environment...
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Python found: %PYTHON_VERSION%
echo.

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not installed!
    echo Please reinstall Python with pip included
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully
echo.

REM Initialize database if it doesn't exist
if not exist "instance\bootcamp.db" (
    echo Initializing database...
    python init_db.py
    if errorlevel 1 (
        echo ERROR: Failed to initialize database
        pause
        exit /b 1
    )
    echo Database initialized successfully
) else (
    echo Database already exists
)
echo.

echo ==========================================
echo   Starting ML Bootcamp Server...
echo ==========================================
echo.
echo   Open your browser and navigate to:
echo   http://localhost:5000
echo.
echo   Press Ctrl+C to stop the server
echo.
echo ==========================================
echo.

REM Start the Flask application
python app.py
