@echo off
title E-Commerce Analytics Pipeline
color 0A

echo ============================================
echo   E-COMMERCE DATA ANALYTICS PIPELINE
echo ============================================
echo.

echo [1/3] Cleaning data and building star schema...
python data_cleaning.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in data_cleaning.py & pause & exit )

echo.
echo [2/3] Running EDA and statistical tests...
python python_analytics.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in python_analytics.py & pause & exit )

echo.
echo [3/3] Running forecasting and customer clustering...
python predictive_modeling.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in predictive_modeling.py & pause & exit )

echo.
echo [4/4] Opening dashboard in browser...
start "" dashboard.html

echo.
echo ============================================
echo   ALL DONE! Dashboard opened in browser.
echo ============================================
pause
