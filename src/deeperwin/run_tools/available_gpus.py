import re
import subprocess
import time

def _get_gpu_memory_in_use():
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
    except:
        return []
    gpu_memory_in_use = []
    for l in nvidia_smi_output.split("\n"):
        r = re.search(r"([0-9]+)W *\/ *([0-9]+)W \| *([0-9]+)MiB *\/ *([0-9]+)MiB", l)
        if r is not None:
            gpu_memory_in_use.append(int(r.group(3)))
    return gpu_memory_in_use

def assign_free_GPU_ids(n_gpus=1, sleep_seconds=0):
    MAXIMUM_USED_GPU_MEMORY = 10 # in MiB
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    used_gpu_memory = _get_gpu_memory_in_use()
    ind_free = [i for i,memory in enumerate(used_gpu_memory) if memory <= MAXIMUM_USED_GPU_MEMORY]

    if len(ind_free) < n_gpus:
        return ""
    else:
        ind_free = ind_free[:n_gpus]
        return ",".join([str(i) for i in ind_free])
