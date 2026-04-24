@echo off
echo ===================================================
echo 📈 KnightSight: Generating Training Graphs...
echo ===================================================
set PYTHON_PATH=C:\Users\GYANENDRA\Miniconda3\envs\ducks\python.exe
%PYTHON_PATH% scripts/plot_results.py
echo.
echo View your metrics at: docs\visualizations\training_metrics.png
echo ===================================================
pause
