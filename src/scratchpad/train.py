import wandb
import sys
import subprocess

run = wandb.init()
print(f"In train.py: config = {run.config}")
run.finish()

print("Launching actual run...")
subprocess.call(["python", "run.py"])
print("Actual run returned. Quitting train.py")






