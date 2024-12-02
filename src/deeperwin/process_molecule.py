#!/usr/bin/env python3
"""
CLI to process a single molecule.
"""

# pylint: disable=import-outside-toplevel
# We deliberately import jax only after the jax.config options have been set
import copy
import os
import sys
from typing import Dict
import numpy as np
from jax.config import config as jax_config
from deeperwin.configuration import Configuration, build_physical_configs_from_changes
import haiku as hk


def _setup_environment(raw_config: Dict, config: Configuration) -> None:
    # Set environment variable to control jax behaviour before importing jax
    if config.computation.disable_tensor_cores:
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    else:
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
    if config.computation.force_device_count and config.computation.n_local_devices:
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={config.computation.n_local_devices}"

    # update jax config
    jax_config.update("jax_enable_x64", config.computation.float_precision == "float64")
    jax_config.update("jax_default_matmul_precision", "highest")

    import chex

    if config.computation.disable_jit:
        chex.fake_pmap_and_jit().start()

    from deeperwin.utils.multihost import configure_hardware, disable_slave_loggers

    # Configure hardware usage
    configure_hardware(config)

    from deeperwin.loggers import build_dpe_root_logger
    from deeperwin.checkpoints import load_data_for_reuse
    from deeperwin.utils.utils import replicate_across_devices

    root_logger = build_dpe_root_logger(config.logging.basic)
    disable_slave_loggers(root_logger)

    # Set random seed
    if config.computation.rng_seed is None:
        rng_seed = np.random.randint(2**31, size=())
        config.computation.rng_seed = int(replicate_across_devices(rng_seed)[0])
    rng_seed = config.computation.rng_seed
    np.random.seed(rng_seed)

    # Reusing/restarting old run: merge configs and load data
    if config.reuse is not None:
        (
            config,
            params_to_reuse,
            fixed_params,
            mcmc_state,
            opt_state,
            clipping_state,
            phisnet_params,
            map_fixed_params,
        ) = load_data_for_reuse(config, raw_config)
    else:
        params_to_reuse, fixed_params, mcmc_state, opt_state, clipping_state, phisnet_params, map_fixed_params = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    return (
        root_logger,
        rng_seed,
        config,
        params_to_reuse,
        fixed_params,
        mcmc_state,
        opt_state,
        clipping_state,
        phisnet_params,
        map_fixed_params,
    )


def process_single_molecule(config_file: str):
    """
    Function that takes in a config file & runs the experiment using a single molecule/geometry
    :param config_file: path to config file

    Note that imports involving jax (so basically most of our code) can only be imported after the
    jax config flags have been set (this is considered best practice).
    """
    raw_config, config = Configuration.load_configuration_file(config_file)
    config.save("full_config.yml")
    (
        root_logger,
        rng_seed,
        config,
        params_to_reuse,
        fixed_params,
        mcmc_state,
        opt_state,
        clipping_state,
        phisnet_params_to_reuse,
        _,
    ) = _setup_environment(raw_config, config)

    from deeperwin.model import build_log_psi_squared
    from deeperwin.optimization import optimize_wavefunction, pretrain_orbitals, evaluate_wavefunction
    from deeperwin.utils.utils import merge_params
    from deeperwin.loggers import finalize_experiment_run, initialize_training_loggers

    training_loggers = initialize_training_loggers(config, use_wandb_group=False, exp_idx_in_group=None)

    # Expand physical config if necessary, ie populate supercell by repeating primitive unit cell
    config.physical = config.physical.get_expanded_if_supercell()

    # Build wavefunction / initialize model parameters
    log_psi_squared, orbital_func, cache_func, params, fixed_params = build_log_psi_squared(
        config.model, config.physical, config.baseline, fixed_params, rng_seed
    )
    if params_to_reuse:
        params = merge_params(params, params_to_reuse, config.reuse.check_param_count)
    training_loggers.log_param("n_params", hk.data_structures.tree_size(params))
    training_loggers.log_metrics(fixed_params.get("baseline_energies"))

    # STEP 1: Supervised pre-training of wavefunction orbitals
    if config.pre_training and config.pre_training.n_epochs > 0:
        root_logger.info("Starting pre-training of orbitals...")
        params, _, mcmc_state = pretrain_orbitals(
            orbital_func,
            cache_func,
            mcmc_state,
            params,
            fixed_params,
            config.pre_training,
            config.physical,
            config.model,
            rng_seed,
            training_loggers,
        )

    # STEP 2: Unsupervised variational wavefunction optimization
    if (config.optimization.n_epochs > 0) or (config.optimization.n_epochs_prev in config.evaluation.opt_epochs):
        root_logger.info("Starting optimization...")
        mcmc_state, params, opt_state, clipping_state = optimize_wavefunction(
            log_psi_squared,
            cache_func,
            params,
            fixed_params,
            mcmc_state,
            config.optimization,
            config.evaluation,
            config.physical,
            config.model.complex_wf,
            config.model.kfac_register_complex,
            rng_seed,
            training_loggers,
            opt_state,
            clipping_state,
        )

    # STEP 3: Wavefunction evaluation (if not already done as "intermediate" evaluation durding optimization)
    if (
        (config.evaluation.n_epochs > 0)
        and config.evaluation.evaluate_final
        and (config.optimization.n_epochs not in config.evaluation.opt_epochs)
    ):
        root_logger.info("Starting evaluation...")
        _, mcmc_state = evaluate_wavefunction(
            log_psi_squared,
            cache_func,
            None,
            params,
            fixed_params,
            mcmc_state,
            config.evaluation,
            config.physical,
            config.model.complex_wf,
            rng_seed,
            training_loggers,
            config.optimization.n_epochs_total,
        )

    # Finalize run: Close loggers, save logfiles
    finalize_experiment_run(config, training_loggers, params, fixed_params, mcmc_state, opt_state, clipping_state)


def process_multiple_molecules_shared(config_file: str) -> None:
    """
    Function that takes in a config file & runs the experiment using shared molecules
    as defined in the config file.
    In this case all parameters are shared, but optimization is done over a set of different geometries
    :param config_file: path to config file

    Note that imports involving jax (so basically most of our code) can only be imported after the
    jax config flags have been set (this is considered best practice).
    """
    raw_config, config = Configuration.load_configuration_file(config_file)
    config.save("full_config.yml")
    (
        root_logger,
        rng_seed,
        config,
        params_to_reuse,
        fixed_params,
        mcmc_state,
        opt_state,
        clipping_state,
        phisnet_params_to_reuse,
        map_fixed_params,
    ) = _setup_environment(raw_config, config)

    # analytical initialization of orbitals is not supported yet in multiple compound setting
    assert (config.model.orbitals.envelope_orbitals is None) or (
        config.model.orbitals.envelope_orbitals.initialization != "analytical"
    )

    from deeperwin.model import build_log_psi_squared, init_model_fixed_params
    from deeperwin.optimization import optimize_shared_wavefunction, pretrain_orbitals_shared, evaluate_wavefunction
    from deeperwin.utils.utils import merge_params
    from deeperwin.geometries import GeometryDataStore
    from deeperwin.loggers import finalize_experiment_run

    physical_configs = build_physical_configs_from_changes(raw_config["physical"])

    # Create geometry data stores
    geometries_data_stores = []
    for idx, physical_config in enumerate(physical_configs):
        # Initialize geometry data store and log primitive config
        g = GeometryDataStore()
        g.idx = idx
        g.weight = physical_config.weight_for_shared
        config_to_log = copy.deepcopy(config)
        config_to_log.physical = physical_config
        g.init_wave_function_logger(config_to_log)

        # Expand physical config if necessary, and store an original copy (for distortions during training)
        physical_config = physical_config.get_expanded_if_supercell()
        g.physical_config = physical_config
        g.physical_config_original = copy.deepcopy(physical_config)  # TODO Necessary?

        # Init fixed model params - If we reuse a shared run don't initialize new weights
        if map_fixed_params is not None:
            hash = physical_config.comment.split("_")[0]
            g.fixed_params, g.mcmc_state, g.clipping_state = map_fixed_params[hash]
        if g.fixed_params is None:
            g.fixed_params = init_model_fixed_params(config.model, physical_config, config.baseline)
        if (config.optimization.shared_optimization.distortion is not None) and (
            config.optimization.shared_optimization.distortion.init_distortion_age == "random"
        ):
            g.n_opt_epochs_last_dist = np.random.randint(0, config.optimization.shared_optimization.distortion.max_age)
        geometries_data_stores.append(g)
    del physical_configs  # remove (un-expanded) physical configs to avoid accidental access; physical configs should only be accessed via geometries_data_stores

    # Choose uniform distribution for geometries that have no weight specified
    total_weight_not_none = sum([g.weight for g in geometries_data_stores if (g.weight is not None)])
    nr_not_none = len([g.weight for g in geometries_data_stores if (g.weight is not None)])
    for g in geometries_data_stores:
        if g.weight is None:
            g.weight = 1 / len(geometries_data_stores)
        else:
            g.weight = (g.weight / total_weight_not_none) * (nr_not_none / len(geometries_data_stores))

    # Build wavefunction / initialize model
    log_psi_squared, orbital_func, cache_func, params, fixed_params = build_log_psi_squared(
        config.model, [g.physical_config for g in geometries_data_stores], config.baseline, fixed_params, rng_seed
    )

    if params_to_reuse:
        params = merge_params(params, params_to_reuse, config.reuse.check_param_count)
    for g in geometries_data_stores:
        g.wavefunction_logger.loggers.log_param("n_params", hk.data_structures.tree_size(params))
        g.wavefunction_logger.loggers.log_metrics(g.fixed_params.get("baseline_energies"))

    # STEP 1: Supervised pre-training of wavefunction orbitals
    if config.pre_training and config.pre_training.n_epochs > 0:
        root_logger.info("Starting pre-training of orbitals...")
        params, geometries_data_stores = pretrain_orbitals_shared(
            orbital_func,
            cache_func,
            geometries_data_stores,
            mcmc_state,
            params,
            config.pre_training,
            config.model,
            config.optimization.shared_optimization.distortion
            if config.pre_training.use_distortions_for_shared_opt
            else None,
            rng_seed,
        )

    # STEP 2: Unsupervised variational wavefunction optimization
    ema_params = None
    if config.optimization.n_epochs > 0:
        root_logger.info("Starting optimization...")
        params, opt_state, geometries_data_stores, ema_params = optimize_shared_wavefunction(
            log_psi_squared,
            cache_func,
            geometries_data_stores,
            config,
            params,
            rng_seed,
            opt_state,
            clipping_state,
        )

    # STEP 3: Wavefunction evaluation
    has_final_been_evaluated = (
        config.optimization.n_epochs * len(geometries_data_stores) in config.evaluation.opt_epochs
    )
    if (config.evaluation.n_epochs > 0) and config.evaluation.evaluate_final and (not has_final_been_evaluated):
        root_logger.info("Starting evaluation...")
        for idx_geom, geometry in enumerate(geometries_data_stores):
            _, mcmc_state = evaluate_wavefunction(
                log_psi_squared,
                cache_func,
                None,
                params,
                geometry.fixed_params,
                geometry.mcmc_state,
                config.evaluation,
                geometry.physical_config,
                config.model.complex_wf,
                rng_seed,
                geometry.wavefunction_logger.loggers,
                geometry.n_opt_epochs,
                dict(opt_n_epoch=config.optimization.n_epochs, geom_id=idx_geom),
            )

    # Finalize runs
    for g in geometries_data_stores:
        finalize_experiment_run(
            config,
            g.wavefunction_logger.loggers,
            params,
            g.fixed_params,
            g.mcmc_state,
            opt_state,
            g.clipping_state,
            ema_params,
        )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[2] == "--shared":
        process_multiple_molecules_shared(sys.argv[1])
    else:
        process_single_molecule(sys.argv[1])
