import jax
import jax.numpy as jnp
import jax.experimental.loops as loops
import functools
import numpy as np
import matplotlib.pyplot as plt

def adjust_stepsize(stepsize, acceptance_rate):
    target_acceptance_rate = 0.5
    return jax.lax.cond(acceptance_rate < target_acceptance_rate, lambda s: s*0.95, lambda s: s*1.05, stepsize)

def _propose_normal(x, rng_state, stepsize):
    return x + jax.random.normal(rng_state, x.shape) * stepsize, jnp.ones(len(x))

def _propose_langevin(x, rng_state, stepsize, grad_func):
    g_x = grad_func(x)
    x_new = x + g_x * stepsize + jnp.sqrt(2*stepsize)*jax.random.normal(rng_state, x.shape)
    g_x_new = grad_func(x_new)
    d_fwd = jnp.sum((x_new - x - g_x*stepsize)**2, axis=-1)
    d_rev = jnp.sum((x - x_new - g_x_new*stepsize)**2, axis=-1)
    q_ratio = (d_fwd - d_rev)/(4.0*stepsize)
    return x_new, q_ratio

def _propose_cauchy(x, rng_state, stepsize):
    return x + jax.random.cauchy(rng_state, x.shape) * stepsize

def make_mcmc_step(func, x, f_x, stepsize, rng_state, proposal_func):
    rng_state, subkey = jax.random.split(rng_state)
    x_new, q_ratio = proposal_func(x, subkey, stepsize)
    f_x_new = func(x_new)
    p_accept = jnp.exp(f_x_new - f_x + q_ratio)

    rng_state, subkey = jax.random.split(rng_state)
    thr_accept = jax.random.uniform(subkey, [len(x)])
    do_accept = p_accept > thr_accept

    f_x_new = jnp.where(do_accept, f_x_new, f_x)
    n_dimensions_x = len(x.shape) - 1
    x_new = jnp.where(do_accept.reshape([-1] + [1]*n_dimensions_x), x_new, x)

    stepsize = adjust_stepsize(stepsize, jnp.mean(do_accept))
    return x_new, f_x_new, stepsize, rng_state

@functools.partial(jax.jit, static_argnums=(0, 3))
def run_mcmc_steps(func, mcmc_state, n_steps, proposal_func):
    loop_body = lambda i, state: make_mcmc_step(func, *state, proposal_func)
    return jax.lax.fori_loop(0, n_steps, loop_body, mcmc_state)


def init_mcmc(x, func_values, proposal, grad_func=None):
    x = jnp.array(x)
    f_x = jnp.array(func_values)
    stepsize = jnp.array(1e-2)
    rng_state = jax.random.PRNGKey(0)
    if proposal == 'normal':
        proposal_func = _propose_normal
    elif proposal == 'langevin':
        proposal_func = functools.partial(_propose_langevin, grad_func=grad_func)
    else:
        raise ValueError("Unknown proposal type")
    return (x, f_x, stepsize, rng_state), proposal_func

if __name__ == '__main__':
    def log_p_func(x):
        peaks = jnp.array([[-1,0],
                          [1,0],
                           [0, 3],
                           [0, -3]], dtype=float)
        weights = jnp.array([1.0, 1.0, 0.5, 0.5])
        sigmas = jnp.array([0.5, 0.5, 0.1, 0.1])
        r = jnp.linalg.norm(x[..., np.newaxis, :] - peaks, axis=-1)
        p = jnp.sum(jnp.exp(-(r/sigmas)**2)*weights, axis=-1)
        return jnp.log(p)

    def target_func(x):
        return jnp.prod(jnp.sin(x), axis=-1)

    grad_func = jax.vmap(jax.grad(log_p_func))
    batch_size = 512
    dims = 2
    burn_in_steps = 600
    inter_steps = 2

    x0 = jnp.array(np.random.randn(batch_size, dims))
    f0 = log_p_func(x0)

    proposals = ['normal', 'langevin']
    x_eq = []
    distances = []
    E_eval = []
    for proposal in proposals:
        print(proposal)
        state, proposal_func = init_mcmc(x0, f0, proposal, grad_func if proposal == 'langevin' else None)
        state1 = run_mcmc_steps(log_p_func, state, burn_in_steps, proposal_func) # burn-in
        x_eq.append(state1[0])
        state2 = make_mcmc_step(log_p_func, *state1, proposal_func)
        dist = jnp.linalg.norm(state2[0]-state1[0], axis=-1)
        distances.append(dist)
        evaluations = []
        for n in range(400):
            state = run_mcmc_steps(log_p_func, state, inter_steps, proposal_func)
            evaluations.append(jnp.mean(target_func(state[0])))
        E_eval.append(evaluations)

    plt.close("all")
    fig, axes = plt.subplots(2,2, figsize=(12,7))
    for name, x, dist, E in zip(proposals, x_eq, distances, E_eval):
        axes[0][0].scatter(x[:,0], x[:,1], label=name, alpha=0.5)
        bins = np.linspace(0, 3.0, 50)
        axes[0][1].hist(dist, label=name, bins=bins, alpha=0.5)
        axes[1][0].plot(E, label=name + f" ({np.mean(E):.4f} +-{np.std(E):.4f}")
    axes[0][0].axis("equal")
    axes[0][0].legend()
    axes[1][0].grid()
    axes[1][0].legend()





