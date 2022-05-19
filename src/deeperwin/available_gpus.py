import re
import subprocess
import random

def _get_gpu_memory_in_use():
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
    except:
        print("nvidia-smi not available")
        return []
    gpu_memory_in_use = []
    for l in nvidia_smi_output.split("\n"):
        r = re.search(r"([0-9]+)W *\/ *([0-9]+)W \| *([0-9]+)MiB *\/ *([0-9]+)MiB", l)
        if r is not None:
            gpu_memory_in_use.append(int(r.group(3)))
    return gpu_memory_in_use

def get_free_GPU_id(require_gpu=False):
    used_gpu_memory = _get_gpu_memory_in_use()
    n_gpus = len(used_gpu_memory)

    if n_gpus == 0:
        if require_gpu:
            raise OSError("Could not detect any GPU using nvidia-smi")
        print("Could not detect any GPU using nvidia-smi! Selecting GPU0.")
        return "0"
    elif n_gpus == 1:
        print("Only 1 GPU available. Selecting GPU0.")
        return "0"
    else:
        ind_free = [i for i in range(n_gpus) if used_gpu_memory[i] <= 10]
        if len(ind_free) == 0:
            if require_gpu:
                raise OSError("No free GPU is available")
            print("No GPUs are free. Selecting GPU0.")
            return "0"
        elif len(ind_free) == 1:
            print(f"Exactly 1 GPU is free. Selecting GPU{ind_free[0]}.")
            return str(ind_free[0])
        else:
            ind_selected = ind_free[random.randint(0, len(ind_free)-1)]
            print(f"Found {n_gpus} GPUs, {len(ind_free)} of which are free. Selecting GPU{ind_selected}")
            return str(ind_selected)
