import subprocess
import time
import torch


def check_gpu_thermal_and_rest(max_temp_threshold=85, cooldown_seconds=30):
    """Queries nvidia-smi and pauses execution if the GPU is too hot."""
    try:
        # Query the exact real-time temperature from nvidia-smi
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        current_temp = int(result.strip().split("\n")[0])

        if current_temp >= max_temp_threshold:
            # 1. Clear GPU memory cache and synchronize operations
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # 2. Complete sleep cycle to let hardware heat dissipate
            time.sleep(cooldown_seconds)

    except Exception as e:
        # Failsafe if nvidia-smi is temporarily busy
        pass
