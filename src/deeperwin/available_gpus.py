import time
import re
import subprocess
import random

SLEEP_TIME_FOR_CONFLICT_RESOLUTION = 30 # seconds

def _get_gpu_memory_in_use():
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
    except FileNotFoundError:
        print("nvidia-smi not available")
        return []
    gpu_memory_in_use = []
    for l in nvidia_smi_output.split("\n"):
        r = re.search(r"([0-9]+)W *\/ *([0-9]+)W \| *([0-9]+)MiB *\/ *([0-9]+)MiB", l)
        if r is not None:
            gpu_memory_in_use.append(int(r.group(3)))
    return gpu_memory_in_use

def get_free_GPU_id():
    used_gpu_memory = _get_gpu_memory_in_use()
    n_gpus = len(used_gpu_memory)

    if n_gpus == 0:
        print("Could not detect any GPU using nvidia-smi! Selecting GPU0.")
        return "0"
    elif n_gpus == 1:
        print("Only 1 GPU available. Selecting GPU0.")
        return "0"
    else:
        ind_free = [i for i in range(n_gpus) if used_gpu_memory[i] == 0]
        print(f"Found {n_gpus} GPUs, {len(ind_free)} of which are free.")
        if len(ind_free) == 0:
            print("No GPUs are free. Selecting GPU0.")
            return "0"
        elif len(ind_free) == 1:
            print(f"Exactly 1 GPU is free. Selecting GPU{ind_free[0]}.")
            return str(ind_free[0])
        else:
            t_sleep = random.randint(1, SLEEP_TIME_FOR_CONFLICT_RESOLUTION)
            print(f"Multiple GPUs are free: Sleeping for randomly chosen time, to avoid collisions: {t_sleep} sec")
            time.sleep(t_sleep)

            used_gpu_memory = _get_gpu_memory_in_use()
            n_gpus = len(used_gpu_memory)
            ind_free = [i for i in range(n_gpus) if used_gpu_memory[i] == 0]
            print(f"Found {n_gpus} GPUs, {len(ind_free)} of which are free.")
            if len(ind_free) == 0:
                print("Selecting GPU0")
                return "0"
            else:
                print(f"Selecting GPU{ind_free[-1]}")
                return str(ind_free[-1])