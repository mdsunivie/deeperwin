import wandb
import numpy as np
import time

default_config = dict(param1=1.0, param2=2.0)
run = wandb.init(config=default_config)
print(f"In run.py: config = {run.config}")

for epoch in range(50):
    time.sleep(0.05)
    metric = np.exp(-epoch/50)
    run.log(dict(error_plus_2_stdev=metric), step=epoch)
run.summary['summ_value'] = 17.4