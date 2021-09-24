import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

rng_key = jax.random.PRNGKey(1234)
true_mean = 80.1234567
float_precision_boundary = true_mean / 2**24
sigma = 1e-3
repetitions = 100
N_values = np.array(np.logspace(2, 6.5, 15), int)
error_values = []

for N in N_values:
    rng_key, subkey = jax.random.split(rng_key)
    E = jax.random.normal(subkey, [N, repetitions]) * sigma + true_mean
    E_mean = jnp.mean(E, axis=0)
    error_values.append(jnp.mean(jnp.abs(E_mean - true_mean)))

# error_values = np.array(error_values, np.float64)

std_error = sigma / np.sqrt(N_values)
plt.close("all")
plt.loglog(N_values, error_values, label="Error of mean")
plt.loglog(N_values, std_error, label="Expected std. error")
plt.axhline(float_precision_boundary, color='k', label='float32 precision')
plt.legend()
plt.xlabel("Nr of samples")
plt.ylabel("Error of the mean")

