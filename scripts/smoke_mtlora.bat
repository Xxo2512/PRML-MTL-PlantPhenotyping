@echo off
REM Windows 本地 MTLoRA smoke 脚本 (实时进度、不缓冲输出)
REM 用法: scripts\smoke_mtlora.bat
REM 预期: 启动 ~30s, 每步 ~2s, 8 步 ~50s 内输出 PASSED

setlocal
set HF_ENDPOINT=https://hf-mirror.com
set PYTHONUNBUFFERED=1

cd /d "%~dp0\.."

echo === [1/2] vanilla 4-task smoke (8 step, num_workers=0, bs=2) ===
python -u train.py ^
    --config configs/method/vanilla.yaml ^
    --steps 8 --tag vanilla_smoke --no_save ^
    --num_workers 0 --batch_per_task 2 --log_every 1
if errorlevel 1 goto :error

echo.
echo === [2/2] MTLoRA 4-task smoke (8 step, num_workers=0, bs=2) ===
python -u train.py ^
    --config configs/method/mtlora.yaml ^
    --steps 8 --tag mtlora_smoke --no_save ^
    --num_workers 0 --batch_per_task 2 --log_every 1
if errorlevel 1 goto :error

echo.
echo === ALL SMOKE PASSED ===
goto :eof

:error
echo === SMOKE FAILED ===
exit /b 1
