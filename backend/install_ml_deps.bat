@echo off
echo ======================================================================
echo Installing ML/AI Dependencies for ReddyGo Platform
echo ======================================================================
echo.

echo Step 1/3: Installing Python packages...
pip install -r requirements.txt

echo.
echo Step 2/3: Verifying installation...
python verify_ml_setup.py

echo.
echo ======================================================================
echo Python packages installed!
echo.
echo Next Steps:
echo   1. Install Docker Desktop (if not installed)
echo   2. Run: docker run -d -p 6333:6333 --name reddygo-qdrant qdrant/qdrant
echo   3. Install Ollama from: https://ollama.ai/download
echo   4. Run: ollama pull llama3.1:8b
echo   5. Re-run: python verify_ml_setup.py
echo ======================================================================
pause
