from deeperwin.checkpoints import load_from_file
from deeperwin.model import build_log_psi_squared, build_log_psi_squared_baseline_model
from deeperwin.configuration import Configuration
import yaml
import numpy as np
import matplotlib.pyplot as plt

import jax
from jax import numpy as jnp


directory = '/Users/leongerard/Desktop/JAX_DeepErwin/data/he_test/'
df = load_from_file(directory + "results.bz2")

with open(directory + "full_config.yml") as f:
    raw_config = yaml.safe_load(f)
config = Configuration.parse_obj(raw_config)



#%%
trainable_params, fixed_params = df['weights']['trainable'], df['weights']['fixed']
mcmc = df['weights']['mcmc']
config.model.distance_feature_powers = [-2]
log_psi_func, _, _ = build_log_psi_squared(config.model, config.physical)
#log_psi_func, _, _ = build_log_psi_squared_baseline_model(config.model.baseline, config.physical)

psi_func_simplified = lambda r: log_psi_func(r, mcmc.R, mcmc.Z, trainable_params, fixed_params)

grad_psi_func = lambda r: jax.grad(psi_func_simplified, argnums=0)(r)
grad_psi_func_batched = jax.vmap(grad_psi_func, in_axes=(0))
#%%
# prepare data
batch_size = 2000

r = np.ones([batch_size, 2, 3])

radius = 1e-1
sphere = lambda theta, phi: (radius*np.sin(theta)*np.cos(phi), radius*np.sin(theta)*np.sin(phi), 0.5 + radius*np.cos(theta))

eps = 1e-7
theta_values = np.random.uniform(0+eps, np.pi -eps, batch_size)
phi_values = np.random.uniform(0+eps, 2*np.pi -eps, batch_size)

r[:, 0, 0] = r[:, 0, 0] * 0.0
r[:, 0, 1] = r[:, 0, 1] * 0.0
r[:, 0, 2] = r[:, 0, 2] * 0.5
for i in range(batch_size):
    r[i, 1, :] = sphere(theta_values[i], phi_values[i])

# plt.close()
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(r[:, 1, 0], r[:, 1, 1], r[:, 1, 2], marker='o')
# ax.scatter(0, 0, 0.5, marker='o', color='r')

# #%%
grad_results = grad_psi_func_batched(r)
grad_results = jnp.reshape(grad_results, (batch_size, -1))
norm_r = np.linalg.norm(r[:, 0, :] - r[:, 1, :], axis=1)

transformation = np.zeros([batch_size, 6])
transformation[:, 0] = (r[:, 0, 0] - r[:, 1, 0])/ norm_r
transformation[:, 1] = (r[:, 0, 1] - r[:, 1, 1])/ norm_r
transformation[:, 2] = (r[:, 0, 2] - r[:, 1, 2])/ norm_r

transformation[:, 3] = (r[:, 1, 0] - r[:, 0, 0])/ norm_r
transformation[:, 4] = (r[:, 1, 1] - r[:, 0, 1])/ norm_r
transformation[:, 5] = (r[:, 1, 2] - r[:, 0, 2])/ norm_r

grad_psi_distance = np.sum(grad_results*transformation, axis=1)
print("Grad psi at r = 0", 0.5*np.mean(grad_psi_distance))
#print("Psi at r = 0: ", np.mean(np.exp(0.5*psi_func_simplified(r))))

#%%

batch_size = 1000
r = np.ones([batch_size, 2, 3])

radius = 1e-1

sphere = lambda theta, phi: (radius * np.sin(theta)*np.cos(phi), radius*np.sin(theta)*np.sin(phi), radius*np.cos(theta))

theta_values = np.random.uniform(0+eps, np.pi -eps, batch_size)
phi_values = np.random.uniform(0+eps, 2*np.pi -eps, batch_size)

r[:, 0, 0] = r[:, 0, 0] * 0.0
r[:, 0, 1] = r[:, 0, 1] * 0.0
r[:, 0, 2] = r[:, 0, 2] * 0.5
for i in range(batch_size):
    r[i, 1, :] = sphere(theta_values[i], phi_values[i])

#%%

grad_results = grad_psi_func_batched(r)
grad_results = jnp.reshape(grad_results, (batch_size, -1))

norm_r = np.linalg.norm(r[:, 1, :], axis=1)
transformation = np.zeros([batch_size, 6])
# transformation[:, 3] = norm_r/ r[:, 1, 0]
# transformation[:, 4] = norm_r/ r[:, 1, 1]
# transformation[:, 5] = norm_r/ r[:, 1, 2]

transformation[:, 3] = r[:, 1, 0]/ norm_r
transformation[:, 4] = r[:, 1, 1]/ norm_r
transformation[:, 5] = r[:, 1, 2]/ norm_r
#print("Psi at rR = 0: ", np.mean(psi_func_simplified(r)))

grad_psi_distance = np.sum(grad_results*transformation, axis=1)
print("Grad psi / psi at rR = 0", 0.5*np.mean(grad_psi_distance))
plt.hist(grad_psi_distance, bins=np.linspace(-20,20,100))

#%%

N_samples = 1000
r = np.ones([N_samples, config.physical.n_electrons, 3])
r[:, 0, 0] = r[:, 0, 0] * 0.0
r[:, 0, 1] = r[:, 0, 1] * 0.0
r[:, 0, 2] = r[:, 0, 2] * 5

r_plot = np.linspace(-0.0001, 0.0001, N_samples)
# r_plot = np.linspace(-np.pi,np.pi, N_samples)
# r[:, 1, 0] = np.sin(r_plot)*0.5
# r[:, 1, 1] = r[:, 1, 1] * 0.0
# r[:, 1, 2] = np.cos(r_plot)*0.5
r[:, 1, 1] = 0
r[:, 1, 2] = 0
r[:, 1, 0] = r_plot
r = jnp.array(r)

R, Z = jnp.array(config.physical.R), jnp.array(config.physical.Z)
print("Calculating log_psi_sqr...")
# log_sqr_baseline = log_psi_squared_baseline(r, R, Z, empty_trainable_params, params_fixed)
log_sqr = psi_func_simplified(r)
grad = grad_psi_func_batched(r)
plt.close()
plt.plot(r_plot, log_sqr, label='Baseline: No NNs', ls='--')
plt.plot(r_plot[1:], 0.5*np.diff(log_sqr)/(r_plot[1] - r_plot[0]))

plt.grid()