@echo off
title Hybrid Search RAG Pipeline
echo ============================================================
echo   Starting Hybrid Search RAG Pipeline...
echo   Opening website at http://localhost:8000
echo ============================================================
echo.

cd /d "N:\hybrid-search-rag-pipeline"

:: Wait 1 second then open the browser
start "" http://localhost:8000

:: Start the unified FastAPI backend & web frontend server
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

pause
