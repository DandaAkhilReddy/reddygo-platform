#!/bin/bash

# ML Bootcamp - Quick Start Script
# This script will set up and run the ML bootcamp in one command

echo "=========================================="
echo "  ML Bootcamp - Quick Start"
echo "  Setting up your learning environment..."
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null
then
    echo "❌ Error: Python is not installed!"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null
then
    PYTHON_CMD=python3
    PIP_CMD=pip3
else
    PYTHON_CMD=python
    PIP_CMD=pip
fi

echo "✅ Python found: $($PYTHON_CMD --version)"
echo ""

# Check if pip is installed
if ! command -v $PIP_CMD &> /dev/null
then
    echo "❌ Error: pip is not installed!"
    echo "Please install pip or reinstall Python with pip included"
    exit 1
fi

echo "📦 Installing dependencies..."
$PIP_CMD install -r requirements.txt --quiet
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo ""

# Initialize database if it doesn't exist
if [ ! -f "instance/bootcamp.db" ]; then
    echo "🗄️  Initializing database..."
    $PYTHON_CMD init_db.py
    if [ $? -eq 0 ]; then
        echo "✅ Database initialized successfully"
    else
        echo "❌ Failed to initialize database"
        exit 1
    fi
else
    echo "✅ Database already exists"
fi
echo ""

echo "=========================================="
echo "  🚀 Starting ML Bootcamp Server..."
echo "=========================================="
echo ""
echo "  Open your browser and navigate to:"
echo "  👉 http://localhost:5000"
echo ""
echo "  Press Ctrl+C to stop the server"
echo ""
echo "=========================================="
echo ""

# Start the Flask application
$PYTHON_CMD app.py
