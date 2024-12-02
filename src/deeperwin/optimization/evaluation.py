"""
Logic for wavefunction evaluation.
"""

import logging
from typing import Tuple, Optional, Dict
import functools
import jax
import jax.numpy as jnp
import numpy as np
from deeperwin.configuration import EvaluationConfig, PhysicalConfig
from deeperwin.hamiltonian import build_local_energy, calculate_forces, calculate_localization_z
from deeperwin.loggers import LoggerCollection, WavefunctionLogger
from deeperwin.mcmc import MCMCState, MetropolisHastingsMonteCarlo
from deeperwin.utils.utils import (
    pmean,
    psum,
    replicate_across_devices,
    estimate_autocorrelation_time,
    get_distance_matrix,
)
from deeperwin.utils.periodic import project_into_first_unit_cell

LOGGER = logging.getLogger("dpe")


@functools.partial(
    jax.pmap,
    axis_name="devices",
    static_broadcasted_argnums=(
        1,
        2,
        3,
    ),
)
def get_density(r_el, r_min, r_max, n_grid, lattice=None):
    """Compute densities from electron coordinates.

    Args:
        r_el (jnp.ndarray): electron coordinates of shape [batch_size, n_el, 3]
        rmin (jnp.ndarray): minimum coordinate of shape [3]
        rmax (jnp.ndarray): maximum coordinate of shape [3]
        n_grid (jnp.ndarray[int]): number of grid points of shape [3]
    """
    r_min = np.array(r_min)
    r_max = np.array(r_max)
    n_grid = np.array(n_grid, int)
    r_center = (r_max + r_min) / 2

    if lattice is not None:
        r_el -= r_center
        r_el = project_into_first_unit_cell(r_el, lattice, around_origin=True)
        r_el += r_center

    dr = (r_max - r_min) / n_grid
    ind_bin = (r_el - r_min) / dr
    ind_bin = ind_bin.astype(int)

    rho = jnp.zeros(n_grid, dtype=jnp.int32)
    rho = rho.at[ind_bin[..., 0], ind_bin[..., 1], ind_bin[..., 2]].add(1, mode="drop")
    rho = jax.lax.psum(rho, axis_name="devices")
    return rho


@functools.partial(jax.pmap, axis_name="devices", in_axes=(0, None, None, None))
def update_welford_estimates_batched(x, ind_batch, mean_est, var_est):
    """Updates estimates for mean and variance using Welford's algorithm."""
    batch_size = x.shape[0] * jax.device_count()
    batch_mean = pmean(jnp.mean(x, axis=0))
    mean_correction = (batch_mean - mean_est) / (ind_batch + 1)
    new_mean_est = mean_est + mean_correction
    new_var_est = var_est + batch_size * ind_batch * mean_correction**2
    new_var_est += psum(jnp.sum(jnp.abs(x - new_mean_est) ** 2, axis=0))
    return new_mean_est, new_var_est


@functools.partial(jax.pmap, axis_name="devices", in_axes=(0, None))
def get_rho_k(r_el, k_vecs):
    """Compute Fourier transform of electron/spin-density from electron coordinates.

    Args:
        r_el (jnp.ndarray): electron coordinates of shape [batch_size, n_el, 3]
        k_vecs (jnp.ndarray): k-vectors of shape [n_k, 3]
    """
    r_el = r_el[..., None, :, :]  # [batch_size, 1,    n_el, 3]
    k_vecs = k_vecs[:, None, :]  # [            n_k,  1,    3]
    rho_k = jnp.sum(jnp.exp(1j * jnp.sum(r_el * k_vecs, axis=-1)), axis=-1)  # sum over xyz and electrons
    return rho_k  # [batch_size, n_k]


def update_structure_factors(rho_k_batch, data_dict, ind_batch, particle_type):
    rho_k, S_k = data_dict[f"rho_k_{particle_type}"], data_dict[f"S_k_{particle_type}"]
    rho_k, S_k = update_welford_estimates_batched(rho_k_batch, ind_batch, rho_k, S_k)
    data_dict[f"rho_k_{particle_type}"] = rho_k[0]
    data_dict[f"S_k_{particle_type}"] = S_k[0]


def evaluate_wavefunction(
    log_psi_sqr,
    cache_func,
    get_local_energy_func,
    params,
    fixed_params,
    mcmc_state: MCMCState,
    config: EvaluationConfig,
    phys_config: PhysicalConfig,
    is_complex: bool,
    rng_seed: int,
    loggers: LoggerCollection = None,
    opt_epoch_nr: int = None,
    extra_summary_metrics: Optional[Dict] = None,
):
    # Burn-in MCMC
    rng = jax.random.PRNGKey(rng_seed)
    LOGGER.debug(f"Starting burn-in for evaluation: {config.mcmc.n_burn_in} steps")
    # static_args are n_up, n_down

    mcmc = MetropolisHastingsMonteCarlo(config.mcmc)
    mcmc_state = MCMCState.resize_or_init(mcmc_state, config.mcmc, phys_config, rng)
    mcmc_state = mcmc_state.split_across_devices()
    batch_size = mcmc_state.r.shape[-3]

    spin_state = (phys_config.n_up, phys_config.n_dn)
    n_up = phys_config.n_up
    lattice = np.array(phys_config.periodic.lattice) if phys_config.periodic is not None else None
    lattice_replicated = np.tile(lattice, (mcmc_state.r.shape[0], 1, 1)) if lattice is not None else None

    params, fixed_params = replicate_across_devices((params, fixed_params))
    cache_func_pmapped = jax.pmap(cache_func, axis_name="devices", static_broadcasted_argnums=(1, 2))
    fixed_params["cache"] = cache_func_pmapped(params, *spin_state, *mcmc_state.build_batch(fixed_params))
    mcmc_state = mcmc.run_burn_in(log_psi_sqr, mcmc_state, params, *spin_state, fixed_params)

    if get_local_energy_func is None:
        get_local_energy_func = build_local_energy(
            log_psi_sqr,
            is_complex,
            phys_config.periodic is not None,
            phys_config.periodic is not None and phys_config.periodic.include_heg_background,
            config.forward_lap,
            config.max_batch_size,
        )

    @functools.partial(jax.pmap, axis_name="devices", static_broadcasted_argnums=(2,))
    def get_observables(params, fixed_params, spin_state: Tuple[int], mcmc_state: MCMCState):
        metrics = dict()
        if config.calculate_energies:
            energies = get_local_energy_func(params, spin_state, *mcmc_state.build_batch(fixed_params))
            metrics["E_mean"] = pmean(jnp.nanmean(energies)).real
            metrics["E_var"] = (
                pmean(jnp.nanmean((energies - metrics["E_mean"]) * jnp.conj(energies - metrics["E_mean"])))
            ).real
        if config.forces:
            forces = calculate_forces(
                log_psi_sqr, params, *mcmc_state.build_batch(fixed_params), mcmc_state.log_psi_sqr, config.forces
            )
            metrics["forces_mean"] = pmean(jnp.nanmean(forces, axis=0))
            metrics["forces_var"] = pmean(jnp.nanmean((forces - metrics["forces_mean"]) ** 2, axis=0))
        if phys_config.periodic is not None and config.localization_metric is not None:
            z = pmean(jnp.mean(calculate_localization_z(mcmc_state.r, fixed_params["periodic"].rec), axis=0))
            for dim in range(3):
                metrics[f"loc_z_{dim}_real"] = z[dim].real
                metrics[f"loc_z_{dim}_imag"] = z[dim].imag
        return metrics

    if config.density is not None:
        grid_r_min, grid_r_max = tuple(config.density.r_min), tuple(config.density.r_max)
        grid_n = tuple(config.density.n_grid)
        rho_cubes = {}
        if config.density.calculate_density:
            rho_cubes["1el"] = np.zeros((2, *grid_n), dtype=np.int32)
        if config.density.calculate_pair_density:
            rho_cubes["2el_rel"] = np.zeros((2, 2, *grid_n), dtype=np.int32)

    compute_structure_factors = lattice is not None and (config.structure_factor_grid is not None)
    if compute_structure_factors:
        ordinals = [np.arange(-n, n + 1) for n in config.structure_factor_grid]
        ordinals = np.stack(np.meshgrid(*ordinals), axis=-1).reshape(-1, 3)
        k_vecs_structure_fac = ordinals @ np.linalg.inv(lattice).T * 2 * np.pi
        structure_fac_data = dict(k_vecs=k_vecs_structure_fac)
        for particle_type in ["el", "up", "dn", "sp"]:
            structure_fac_data[f"rho_k_{particle_type}"] = np.zeros(len(k_vecs_structure_fac), dtype=np.complex64)
            structure_fac_data[f"S_k_{particle_type}"] = np.zeros(len(k_vecs_structure_fac), dtype=float)

    # Evaluation loop
    wf_logger = WavefunctionLogger(loggers, prefix="eval", smoothing=1.0)
    metrics_history = {}
    for step in range(config.n_epochs):
        mcmc_state = mcmc.run_inter_steps(log_psi_sqr, mcmc_state, params, *spin_state, fixed_params)

        # Compute standard metrics
        metrics = get_observables(params, fixed_params, spin_state, mcmc_state)
        metrics = {k: v[0] for k, v in metrics.items()}  # pick data from first GPU
        if step == 0:  # Allocate memory for metrics history
            for key, value in metrics.items():
                metrics_history[key] = np.zeros((config.n_epochs, *np.array(value).shape), value.dtype)
        for key, value in metrics.items():
            metrics_history[key][step, ...] = value
        mcmc_state_merged = mcmc_state.merge_devices()
        wf_logger.log_step(
            metrics, E_ref=phys_config.E_ref, mcmc_state=mcmc_state_merged, extra_metrics={"opt_epoch": opt_epoch_nr}
        )

        # Compute densities
        if config.density is not None:
            if "1el" in rho_cubes:
                for spin, el_slice in enumerate([slice(0, n_up), slice(n_up, None)]):
                    # Take result from GPU0
                    rho_cubes["1el"][spin, ...] += get_density(
                        mcmc_state.r[..., el_slice, :], grid_r_min, grid_r_max, grid_n, lattice_replicated
                    )[0]
            if "2el_rel" in rho_cubes:
                for spin1, el_slice1 in enumerate([slice(0, n_up), slice(n_up, None)]):
                    for spin2, el_slice2 in enumerate([slice(0, n_up), slice(n_up, None)]):
                        if spin1 == spin2:
                            diff, _ = get_distance_matrix(mcmc_state.r[..., el_slice1, :], full=False)
                        else:
                            diff = mcmc_state.r[..., el_slice1, None, :] - mcmc_state.r[..., None, el_slice2, :]
                        rho_cubes["2el_rel"][spin1, spin2, ...] += get_density(
                            diff, grid_r_min, grid_r_max, grid_n, lattice_replicated
                        )[0]

        # Compute structure factors: Fourier transform of density and the variance of rho(k)
        if compute_structure_factors:
            rho_batches = {}
            rho_batches["up"] = get_rho_k(mcmc_state.r[..., :n_up, :], k_vecs_structure_fac)
            rho_batches["dn"] = get_rho_k(mcmc_state.r[..., n_up:, :], k_vecs_structure_fac)
            rho_batches["el"] = rho_batches["up"] + rho_batches["dn"]
            rho_batches["sp"] = rho_batches["up"] - rho_batches["dn"]
            for particle_type in ["el", "up", "dn", "sp"]:
                update_structure_factors(rho_batches[particle_type], structure_fac_data, step, particle_type)

    # save files only on the master process
    if jax.process_index() == 0:
        if config.density is not None:
            for density_type, rho in rho_cubes.items():
                if opt_epoch_nr is None:
                    density_fname = f"rho_{density_type}.npz"
                else:
                    density_fname = f"rho_{density_type}_{opt_epoch_nr:06d}.npz"
                np.savez(density_fname, rho=rho, r_min=grid_r_min, r_max=grid_r_max, n_grid=grid_n)

        if compute_structure_factors:
            structure_fac_data["S_k_el"] /= config.n_epochs * batch_size * phys_config.n_electrons * jax.device_count()
            structure_fac_data["S_k_sp"] /= config.n_epochs * batch_size * phys_config.n_electrons * jax.device_count()
            structure_fac_data["S_k_up"] /= config.n_epochs * batch_size * phys_config.n_up * jax.device_count()
            structure_fac_data["S_k_dn"] /= config.n_epochs * batch_size * phys_config.n_dn * jax.device_count()

            geom_id = None
            if (extra_summary_metrics is not None) and ("geom_id" in extra_summary_metrics.keys()):
                geom_id = f"{int(extra_summary_metrics['geom_id']):04}/"

            if opt_epoch_nr is None:
                structure_fac_fname = f"{geom_id if geom_id is not None else ''}structure_factors.npz"
            else:
                structure_fac_fname = (
                    f"{geom_id if geom_id is not None else ''}structure_factors_{opt_epoch_nr:06d}.npz"
                )
            np.savez(structure_fac_fname, **structure_fac_data)

    # Average computed metrics over steps
    summary = extra_summary_metrics or {}
    for key, history in metrics_history.items():
        summary[key] = np.nanmean(history, axis=0)
        N_samples = len(history)
        summary[key + "_sigma"] = np.nanstd(history, axis=0) / np.sqrt(N_samples)
        if history.ndim == 1:
            tau_autocorr = estimate_autocorrelation_time(history)
            summary[key + "_sigma_corr"] = summary[key + "_sigma"] * np.sqrt(2 * tau_autocorr)

    # Compute absolute localization from individual components
    if "loc_z_0_real" in summary:
        for dim in range(3):
            summary[f"loc_abs_{dim}"] = np.sqrt(summary[f"loc_z_{dim}_real"] ** 2 + summary[f"loc_z_{dim}_imag"] ** 2)
            summary[f"loc_abs_{dim}_sigma"] = np.sqrt(
                summary[f"loc_z_{dim}_real_sigma"] ** 2 + summary[f"loc_z_{dim}_imag_sigma"] ** 2
            )
            summary[f"loc_abs_{dim}_sigma_corr"] = np.sqrt(
                summary[f"loc_z_{dim}_real_sigma_corr"] ** 2 + summary[f"loc_z_{dim}_imag_sigma_corr"] ** 2
            )
        ind_min = np.argmin([summary[f"loc_abs_{dim}"] for dim in range(3)])
        summary["loc_abs_min"] = summary[f"loc_abs_{ind_min}"]
        summary["loc_abs_min_sigma"] = summary[f"loc_abs_{ind_min}_sigma"]
        summary["loc_abs_min_sigma_corr"] = summary[f"loc_abs_{ind_min}_sigma_corr"]

    wf_logger.log_summary(phys_config.E_ref, opt_epoch_nr, summary)
    return wf_logger.history, mcmc_state_merged
