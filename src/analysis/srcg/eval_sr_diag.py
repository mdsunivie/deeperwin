#%%
from deeperwin.checkpoints import load_run
from deeperwin.model.wavefunction import build_log_psi_squared
import jax
import jax.numpy as jnp
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default="grads.npy")
    parser.add_argument("fnames", nargs="+", type=str)
    args = parser.parse_args()
    batch_size = args.batch_size

    output_data = {}
    for ind_checkpoint, fname in enumerate(args.fnames):
        print(f"Processing checkpoint {fname}")
        data = load_run(fname)

        if ind_checkpoint == 0:
            log_psi_sqr_full = build_log_psi_squared(data.config.model, 
                                data.config.physical,
                                data.fixed_params,
                                0,
                                None,
                                None,
                                None)[0]

            n_el, n_up, R, Z = data.config.physical.get_basic_params()
            n_dn = n_el - n_up
            def log_psi_sqr(params, r):
                return log_psi_sqr_full(params, 
                                        data.config.physical.n_up, 
                                        data.config.physical.n_dn,
                                        r,
                                        R,
                                        Z,
                                        data.fixed_params)

            @jax.jit
            def per_example_grad(params, r):
                return jax.vmap(jax.grad(log_psi_sqr), in_axes=(None, 0), out_axes=0)(params, r)

        grads_squared = jax.tree_util.tree_map(jnp.zeros_like, data.params)
        grads_mean = jax.tree_util.tree_map(jnp.zeros_like, data.params)
        n_batches = data.mcmc_state.r.shape[0] // batch_size
        n_samples = n_batches * batch_size
        r_all = data.mcmc_state.r[:n_samples].reshape((n_batches, batch_size, -1, 3))
        for i, r_batch in enumerate(r_all):
            grads = per_example_grad(data.params, r_batch)
            grads_squared = jax.tree_util.tree_map(lambda g, x: g + jnp.sum(x**2, axis=0), grads_squared, grads)
            grads_mean = jax.tree_util.tree_map(lambda g, x: g + jnp.sum(x, axis=0), grads_mean, grads)
        grads_squared = jax.tree_util.tree_map(lambda g: np.array(g / n_samples), grads_squared)
        grads_mean = jax.tree_util.tree_map(lambda g: np.array(g / n_samples), grads_mean)
        output_data[fname] = dict(grads_squared=grads_squared, grads_mean=grads_mean)

    print(f"Saving data to {args.output}")
    np.save(args.output, output_data)




