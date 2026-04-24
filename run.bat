@echo off
echo ===================================================
echo 🚗 KnightSight EdgeVision: Starting Dashboard...
echo ===================================================
set PYTHON_PATH=C:\Users\GYANENDRA\Miniconda3\envs\ducks\python.exe
%PYTHON_PATH% -m streamlit run app.py
pause
