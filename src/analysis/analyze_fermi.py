from deeperwin.checkpoints import load_from_file
from deeperwin.model import build_log_psi_squared, build_log_psi_squared_baseline_model
from deeperwin.configuration import Configuration
import yaml
import numpy as np
import matplotlib.pyplot as plt

import jax
from jax import numpy as jnp


directory = '/Users/leongerard/Desktop/JAX_DeepErwin/data/fermi/dump_weights/'
chkpt = ['chkpt000050', 'chkpt001000']

with open(directory + "full_config.yml") as f:
    raw_config = yaml.safe_load(f)
config = Configuration.parse_obj(raw_config)

df = load_from_file(directory + chkpt[0] + "/results.bz2")
trainable_params, fixed_params = df['weights']['trainable'], df['weights']['fixed']

df = load_from_file(directory + chkpt[1] + "/results.bz2")
trainable_params2, fixed_params = df['weights']['trainable'], df['weights']['fixed']

embed = trainable_params['embed']
embed2 = trainable_params2['embed']

fac = trainable_params['bf_fac_orbital']
fac2 = trainable_params2['bf_fac_orbital']

# for i in chkpt:
#     df = load_from_file(directory + i + "/results.bz2")
#     trainable_params, fixed_params = df['weights']['trainable'], df['weights']['fixed']