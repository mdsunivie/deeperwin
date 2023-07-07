# TODO: refactor to not duplicate as much between shared and independent optimization; potentially formulate independent optimization as shared with a single geometry?
import logging
from typing import Callable, Dict, Optional, Any, Tuple, List
from deeperwin.geometries import GeometryDataStore, distort_geometry
import jax
import jax.numpy as jnp
import numpy as np

from deeperwin.configuration import PreTrainingConfig, ModelConfig, PhysicalConfig, DistortionConfig
from deeperwin.loggers import DataLogger
from deeperwin.mcmc import MetropolisHastingsMonteCarlo, MCMCState
from deeperwin.utils.utils import get_el_ion_distance_matrix, without_cache
from deeperwin.model import evaluate_sum_of_determinants, get_baseline_slater_matrices, init_model_fixed_params
from deeperwin.orbitals import get_baseline_solution, get_sum_of_atomic_exponentials
from deeperwin.optimizers import build_optimizer
from deeperwin.utils.utils import replicate_across_devices, get_from_devices, get_next_geometry_index
from deeperwin.optimization.opt_utils import _run_mcmc_with_cache
from deeperwin.checkpoints import is_checkpoint_required, delete_obsolete_checkpoints

LOGGER = logging.getLogger("dpe")

def _get_orbitals(fixed_params):
    if "pretrain_orbitals" in fixed_params:
        return fixed_params["pretrain_orbitals"]
    elif "orbitals" in fixed_params:
        return fixed_params["orbitals"]
    elif "transferable_atomic_orbitals" in fixed_params:
        return fixed_params["transferable_atomic_orbitals"]["orbitals"]
    else:
        raise ValueError("No orbitals found for pretrianing in fixed params")

def build_pretraining_loss_func(orbital_func, pretrain_config, model_config):
    def loss_func(params, batch, spin_state):
        r, R, Z, fixed_params = batch
        n_up, n_dn = spin_state

        # Calculate HF / CASSCF reference orbitals
        diff_el_ion, dist_el_ion = get_el_ion_distance_matrix(r, R)
        mo_up_ref, mo_dn_ref = get_baseline_slater_matrices(
            diff_el_ion, dist_el_ion, _get_orbitals(fixed_params), model_config.orbitals.determinant_schema
        )

        if pretrain_config.use_only_leading_determinant:
            mo_up_ref = mo_up_ref[..., :1, :, :]
            mo_dn_ref = mo_dn_ref[..., :1, :, :]

        if (pretrain_config.off_diagonal_mode == "exponential") and (model_config.orbitals.determinant_schema != "block_diag"):
            phi_exp = get_sum_of_atomic_exponentials(
                dist_el_ion, exponent=pretrain_config.off_diagonal_exponent, scale=pretrain_config.off_diagonal_scale
            )  # [batch x n_el]
            mo_up_ref = mo_up_ref.at[..., :, :, n_up:].set(phi_exp[..., None, :n_up, None])
            mo_dn_ref = mo_dn_ref.at[..., :, :, :n_up].set(phi_exp[..., None, n_up:, None])

        # Calculate neural net orbitals
        mo_up, mo_dn = orbital_func(params, n_up, n_dn, r, R, Z, without_cache(fixed_params))
        residual_up = mo_up - mo_up_ref
        residual_dn = mo_dn - mo_dn_ref
        if pretrain_config.off_diagonal_mode == "ignore":
            residual_up = residual_up[..., :, :n_up]
            residual_dn = residual_dn[..., :, n_up:]
        return jnp.mean(residual_up ** 2) + jnp.mean(residual_dn ** 2)

    return loss_func


def build_log_psi_sqr_func_for_sampling(orbital_func, pretrain_config, model_config):
    def log_psi_squared_func(params, n_up, n_dn, r, R, Z, fixed_params):
        if pretrain_config.sampling_density == "model":
            mo_up, mo_dn = orbital_func(params, n_up, n_dn, r, R, Z, fixed_params)
        elif pretrain_config.sampling_density == "reference":
            diff_el_ion, dist_el_ion = get_el_ion_distance_matrix(r, R)
            mo_up, mo_dn = get_baseline_slater_matrices(
                diff_el_ion, dist_el_ion, _get_orbitals(fixed_params), model_config.orbitals.determinant_schema
            )
        else:
            raise ValueError(f"Unknown sampling scheme for pre-training: {pretrain_config.sampling_density} ")
        return evaluate_sum_of_determinants(mo_up, mo_dn)

    return log_psi_squared_func


def pretrain_orbitals(
    orbital_func: Callable,
    cache_func: Callable,
    mcmc_state: MCMCState,
    params: Dict,
    fixed_params: Dict,
    pretrain_config: PreTrainingConfig,
    phys_config: PhysicalConfig,
    model_config: ModelConfig,
    rng_seed: int,
    logger: Optional[DataLogger] = None,
    opt_state: Optional[Any] = None,
) -> Tuple[Dict, Any, Any]:
    if model_config.orbitals.baseline_orbitals and (model_config.orbitals.baseline_orbitals.baseline == pretrain_config.baseline):
        LOGGER.debug("Identical CASSCF-config for pre-training and baseline model: Reusing baseline calculation")
    elif model_config.orbitals.transferable_atomic_orbitals:
        LOGGER.debug("Re-using baseline calculation of Transferable Atomic Orbitals for pre-training.")
    else:
        LOGGER.warning(
            "Using different baseline pySCF settings for pre-training and the baked-in baseline model. Calculating new orbitals..."
        )
        n_determinants = 1 if pretrain_config.use_only_leading_determinant else model_config.orbitals.n_determinants
        fixed_params["pretrain_orbitals"], (E_hf, E_casscf) = get_baseline_solution(phys_config, pretrain_config.baseline, n_determinants)
        LOGGER.debug(f"Finished baseline calculation for pretraining: E_HF = {E_hf:.6f}, E_casscf={E_casscf:.6f}")

    loss_func = build_pretraining_loss_func(orbital_func, pretrain_config, model_config)
    log_psi_squared_func = build_log_psi_sqr_func_for_sampling(orbital_func, pretrain_config, model_config)

    # Init MCMC
    rng_mcmc, rng_opt = jax.random.split(jax.random.PRNGKey(rng_seed), 2)
    logging.debug(f"Starting pretraining...")
    mcmc = MetropolisHastingsMonteCarlo(pretrain_config.mcmc)
    mcmc_state = MCMCState.resize_or_init(
        mcmc_state, pretrain_config.mcmc.n_walkers, phys_config, pretrain_config.mcmc.initialization, rng_mcmc
    )
    spin_state = (phys_config.n_up, phys_config.n_dn)

    params, fixed_params, rng_opt = replicate_across_devices((params, fixed_params, rng_opt))
    mcmc_state, fixed_params = _run_mcmc_with_cache(
        log_psi_squared_func, cache_func, mcmc, params, spin_state, mcmc_state, fixed_params, split_mcmc=True, merge_mcmc=False, mode="burnin"
    )

    # Init optimizer
    optimizer = build_optimizer(
        value_and_grad_func=jax.value_and_grad(loss_func),
        opt_config=pretrain_config.optimizer,
        value_func_has_aux=False,
        value_func_has_state=False,
    )
    opt_state = opt_state or optimizer.init(params=params, rng=rng_opt, batch=mcmc_state.build_batch(fixed_params), static_args=spin_state)

    # Pre-training optimization loop
    for n in range(pretrain_config.n_epochs):
        mcmc_state, fixed_params = _run_mcmc_with_cache(
            log_psi_squared_func,
            cache_func,
            mcmc,
            params,
            spin_state,
            mcmc_state,
            fixed_params,
            split_mcmc=False,
            merge_mcmc=False,
            mode="intersteps",
        )
        params, opt_state, stats = optimizer.step(
            params=params, state=opt_state, static_args=spin_state, rng=rng_opt, batch=mcmc_state.build_batch(fixed_params)
        )
        mcmc_state_merged = mcmc_state.merge_devices()

        if logger is not None:
            logger.log_metrics(
                dict(
                    loss=float(stats["loss"].mean()),
                    mcmc_stepsize=float(mcmc_state_merged.stepsize.mean()),
                    mcmc_step_nr=int(mcmc_state_merged.step_nr.mean()),
                ),
                epoch=n,
                metric_type="pre",
            )

        save_checkpoint = is_checkpoint_required(n, pretrain_config.checkpoints) or (n == pretrain_config.n_epochs - 1)
        if save_checkpoint and (logger is not None):
            LOGGER.debug(f"Saving checkpoint n_pre_epoch={n}")
            params_merged, fixed_params_merged, opt_state_merged = get_from_devices((params, fixed_params, opt_state))
            logger.log_checkpoint(
                n, params_merged, fixed_params_merged, mcmc_state_merged, opt_state_merged, None, prefix="pre"
            )
            delete_obsolete_checkpoints(n, pretrain_config.checkpoints, prefix="pre")

    # save checkpoint after pre-training
    params, fixed_params, opt_state = get_from_devices((params, fixed_params, opt_state))
    return params, opt_state, mcmc_state_merged


def pretrain_orbitals_shared(
    orbital_func: Callable,
    cache_func: Callable,
    geometries_data_stores: List[GeometryDataStore],
    mcmc_state: Optional[MCMCState],
    params: Dict[str, Dict[str, jnp.DeviceArray]],
    pretrain_config: PreTrainingConfig,
    model_config: ModelConfig,
    distortion_config: Optional[DistortionConfig],
    rng_seed: int,
    opt_state: Optional[Any] = None,
    phisnet_model = None,
    N_ions_max = None,
    nb_orbitals_per_Z = None,
) -> Tuple[Dict, Any]:
    # for each geometry set the pretraining orbital targets
    for idx_geom, g in enumerate(geometries_data_stores):
        if model_config.orbitals.baseline_orbitals and (model_config.orbitals.baseline_orbitals.baseline == pretrain_config.baseline):
            g.fixed_params["pretrain_orbitals"] = g.fixed_params["orbitals"]
            LOGGER.debug(f"Geom {idx_geom}: Identical CASSCF-config for pre-training and baseline model: Reusing baseline calculation")
        elif model_config.orbitals.transferable_atomic_orbitals:
            g.fixed_params["pretrain_orbitals"] = g.fixed_params["transferable_atomic_orbitals"]["orbitals"]
            LOGGER.debug(f"Geom {idx_geom}: Re-using baseline calculation of Transferable Atomic Orbitals for pre-training.")
        else:
            LOGGER.warning(
                f"Geom {idx_geom}: Using different baseline pySCF settings for pre-training and the baked-in baseline model. Calculating new orbitals..."
            )
            n_determinants = 1 if pretrain_config.use_only_leading_determinant else model_config.orbitals.n_determinants
            g.fixed_params["pretrain_orbitals"], (E_hf, E_casscf) = get_baseline_solution(
                g.physical_config, pretrain_config.baseline, n_determinants
            )
            LOGGER.debug(f"Geom {idx_geom}: Finished baseline calculation for pretraining: E_HF = {E_hf:.6f}, E_casscf={E_casscf:.6f}")

    loss_func = build_pretraining_loss_func(orbital_func, pretrain_config, model_config)
    log_psi_squared_func = build_log_psi_sqr_func_for_sampling(orbital_func, pretrain_config, model_config)

    # Init MCMC
    logging.debug(f"Starting pretraining...")
    rng_opt = jax.random.PRNGKey(rng_seed)
    mcmc = MetropolisHastingsMonteCarlo(pretrain_config.mcmc)
    params, opt_state, rng_opt = replicate_across_devices((params, opt_state, rng_opt))

    # create MCMC state & run burn in for each geometry
    for idx, g in enumerate(geometries_data_stores):
        logging.debug(f"Running burn-in for geom {idx}")
        g.spin_state = (g.physical_config.n_up, g.physical_config.n_dn)
        g.mcmc_state = MCMCState.resize_or_init(
            mcmc_state,
            pretrain_config.mcmc.n_walkers,
            g.physical_config,
            pretrain_config.mcmc.initialization,
            jax.random.PRNGKey(rng_seed + idx),
        )
        g.fixed_params = replicate_across_devices(g.fixed_params)
        g.mcmc_state, g.fixed_params = _run_mcmc_with_cache(
            log_psi_squared_func,
            cache_func,
            mcmc,
            params,
            g.spin_state,
            g.mcmc_state,
            g.fixed_params,
            split_mcmc=True,
            merge_mcmc=True,
            mode="burnin",
        )

    # Init loss and optimizer
    optimizer = build_optimizer(jax.value_and_grad(loss_func), pretrain_config.optimizer, False, False)
    opt_state = opt_state or optimizer.init(
        params=params,
        rng=rng_opt,
        batch=geometries_data_stores[0].mcmc_state.split_across_devices().build_batch(geometries_data_stores[0].fixed_params),
        static_args=geometries_data_stores[0].spin_state,
    )

    # Pre-training optimization loop
    geometry_permutation = np.asarray(jax.random.permutation(jax.random.PRNGKey(rng_seed), len(geometries_data_stores)))
    for n_epoch in range(pretrain_config.n_epochs * len(geometries_data_stores)):
        n_epoch_for_each_geom = n_epoch // len(geometries_data_stores)

        # Step 1. select next geometry
        next_geometry_index = get_next_geometry_index(n_epoch, geometries_data_stores, "round_robin", None, None, geometry_permutation)
        g = geometries_data_stores[next_geometry_index]

        if distortion_config and g.n_opt_epochs_last_dist >= distortion_config.max_age:
            g.fixed_params, g.clipping_state = get_from_devices((g.fixed_params, g.clipping_state))
            if jax.process_index() == 0:
                E_old = g.fixed_params["baseline_energies"].get("E_hf", np.nan)
                g = distort_geometry(g, distortion_config)
                g.fixed_params = init_model_fixed_params(model_config, 
                                                         g.physical_config, 
                                                         phisnet_model, 
                                                         N_ions_max,
                                                         g.fixed_params['transferable_atomic_orbitals']["orbitals"].atomic_orbitals
                                                         )
                E_new = g.fixed_params["baseline_energies"].get("E_hf", np.nan)
                LOGGER.debug(f"New geometry: geom_id={g.idx}; R_new={g.physical_config.R}; U_new={g.rotation.tolist()}, delta_E={E_new-E_old:.6f}")
            g.fixed_params, g.clipping_state = replicate_across_devices((g.fixed_params, g.clipping_state))

        # Step 2: Split MCMC state across devices and run MCMC intersteps
        g.mcmc_state, g.fixed_params = _run_mcmc_with_cache(
            log_psi_squared_func,
            cache_func,
            mcmc,
            params,
            g.spin_state,
            g.mcmc_state,
            g.fixed_params,
            split_mcmc=True,
            merge_mcmc=False,
            mode="intersteps",
        )

        # Step 3: Optimize wavefunction
        params, opt_state, stats = optimizer.step(
            params, opt_state, static_args=g.spin_state, rng=rng_opt, batch=g.mcmc_state.build_batch(g.fixed_params)
        )

        # Step 4. gather states across devices again
        g.mcmc_state = g.mcmc_state.merge_devices()

        # Step 5. update & log metrics
        g.n_opt_epochs_last_dist += 1
        g.wavefunction_logger.loggers.log_metrics(
            metrics=dict(
                loss=float(stats["loss"].mean()),
                mcmc_stepsize=float(g.mcmc_state.stepsize.mean()),
                mcmc_step_nr=int(g.mcmc_state.step_nr.mean()),
                n_epoch=n_epoch,
                geom_id=next_geometry_index,
            ),
            epoch=n_epoch_for_each_geom,
            metric_type="pre",
        )

    params, opt_state = get_from_devices((params, opt_state))
    for g in geometries_data_stores:
        g.fixed_params = get_from_devices(g.fixed_params)
        g.mcmc_state = g.mcmc_state.merge_devices()
    return params, opt_state
