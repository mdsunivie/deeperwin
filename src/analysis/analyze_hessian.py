import jax.numpy as jnp
from utils import load_from_file
from bfgs import calculate_hvp_by_2loop_recursion
import matplotlib.pyplot as plt
import numpy as np

fname = r'/home/mscherbela/runs/debug/debug_hessian/debug_hessian/chkpt9000.bz2'
data = load_from_file(fname)
s,y,rho = data['weights']['opt'][1]
#%%

graw = (s[-1] - s[-2]) * 1e5
g = calculate_hvp_by_2loop_recursion((s,y,rho), graw)
ratio = g / (graw + 1e-12)

plt.close("all")
plt.hist(ratio, bins=np.linspace(-2,2,1000))


