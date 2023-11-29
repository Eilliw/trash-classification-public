@REM call workon venv && cd path/to/Python/proj &&nvidia-smi daemon &&gpu-data.bat
@REM nvidia-smi daemon
@REM gpustat --no-processes --interval 5
"venv\Scripts\activate.bat" &&gpustat --no-processes --interval 5

@REM nividia-smi daemon -t
