@echo off
setlocal enabledelayedexpansion

echo ================================================================================
echo   Cloud Architecture & FinOps Practice AWS SPOT LIFECYCLE ^& PIPELINE SURVIVAL ENGINE [GP-101]
echo   Reproducible Local-First Execution, FinOps Analytics ^& Decision Platform
echo ================================================================================
echo.

if "%1"=="--test" goto :run_tests

if not exist data\spot_events goto :bootstrap_data
goto :run_demo

:bootstrap_data
echo [Lakehouse] Bootstrapping Hive-partitioned Spot Event Data Lake - 10000 records...
python src/data_generator.py --records 10000
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Data generation failed
    exit /b %ERRORLEVEL%
)
echo.

:run_demo
echo [TUI Demo] Launching Rich Executive Terminal Interface...
python src/interface.py --demo
goto :end

:run_tests
echo [Test Suite] Running 35 Automated Pytest Verifications...
python -m pytest tests/ -v
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Test suite failed
    exit /b %ERRORLEVEL%
)
echo.
echo [Benchmark] Running Quantitative Latency Benchmarks...
python tests/benchmark.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Benchmark failed
    exit /b %ERRORLEVEL%
)
goto :end

:end
echo.
echo ================================================================================
echo   Demonstration Completed Successfully.
echo   For interactive control deck, run: python src/interface.py
echo   To execute full test suite and latency benchmarks, run: run_demo.bat --test
echo ================================================================================
