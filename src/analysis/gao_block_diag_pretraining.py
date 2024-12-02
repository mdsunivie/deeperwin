from deeperwin.checkpoints import load_run
from deeperwin.model.wavefunction import build_log_psi_squared
from deeperwin.model.definitions import NParticles
import jax.numpy as jnp

fnames = [
    "/home/mscherbela/runs/gao/regressions/2023-01-25/debug_Cylco_exponential_0.1_3.0_8000/prechkpt007999.zip",
    "/home/mscherbela/runs/gao/regressions/2023-01-25/ref_2023-01-25_Cyclobutadiene_full_det_4_4000_rep2/prechkpt003999.zip",
    "/home/mscherbela/runs/gao/regressions/2023-01-25/gaosymm_2023-01-25_Cyclobutadiene_full_det_4_4000_rep2/prechkpt003999.zip",
    "/home/mscherbela/runs/gao/regressions/2023-01-25/gaosymm_exp_2023-01-25_Cyclobutadiene_full_det_4_16000_rep2/prechkpt015999.zip",
    "/home/mscherbela/runs/gao/regressions/2023-01-25/debug_Cylco_reference_16000/prechkpt015999.zip"]

for fname in fnames:
    print(fname)
    run_data = load_run(fname, parse_config=True)
    config = run_data.config
    n_up, n_dn = config.physical.n_up, config.physical.n_dn

    log_psi_sqr, orbital_func, params_init, fixed_params = build_log_psi_squared(config.model,
                                                                                 config.physical,
                                                                                 NParticles(config.physical.n_ions,
                                                                                            n_up,
                                                                                            n_dn),
                                                                                 shared_optimization=False,
                                                                                 fixed_params=run_data.fixed_params,
                                                                                 rng_seed=0,
                                                                                 )
    params = run_data.params
    # params = params_init

    mo_up, mo_dn = orbital_func(params, n_up, n_dn, *run_data.mcmc_state.build_batch(run_data.fixed_params))
    up_diag = mo_up[..., :, :n_up]
    up_offdiag = mo_up[..., :, n_up:]
    dn_diag = mo_dn[..., :, n_up:]
    dn_offdiag = mo_dn[..., :, :n_up]

    for mo, label in zip([up_diag, dn_diag, up_offdiag, dn_offdiag],
                         ["up_diag", "dn_diag", "up_offdiag", "dn_offdiag"]):
        print(
            f"{label:<10}: {jnp.mean(mo): 6.4f} +- {jnp.std(mo): 6.4f}; [min={jnp.min(mo): 10.4f}, max={jnp.max(mo): 10.4f}]")
    print("\n\n")
