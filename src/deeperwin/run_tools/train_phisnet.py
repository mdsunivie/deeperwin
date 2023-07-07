import pickle
import numpy as np
import os
import sys
import functools
import math
import re

# Set environment variable to control jax behaviour before importing jax
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["JAX_ENABLE_X64"] = "True"

import jax.nn
import jax.numpy as jnp
import optax
import haiku as hk
import chex

from deeperwin.configuration import PhisNetTrainingConfiguration
from deeperwin.orbitals import _get_orbital_mapping, _get_all_basis_functions, align_orbitals, get_cosine_dist
from deeperwin.model.ml_orbitals.ml_orbitals import E3PhisNet, solve_roothaan_equations
from deeperwin.utils.utils import get_distance_matrix, get_param_size_summary
from deeperwin.checkpoints import RunData, save_run, load_run
from deeperwin.optimizers import build_optax_optimizer
from deeperwin.loggers import (
    build_dpe_root_logger,
    WavefunctionLogger,
    LoggerCollection,
)
from deeperwin.utils.setup_utils import initialize_training_loggers
import e3nn_jax as e3nn

# NB_ORBITALS_PER_Z = {8: 14, 1: 5}
# NB_ORBITALS_PER_Z = {6: 13, 7: 13, 8: 13, 1: 3}
# TODO: infer from basis-set
# NB_ORBITALS_PER_Z = {6: 5, 7: 5, 8: 5, 1: 1}


class PatienceLRSchedule:
    def __init__(self, initial_lr, patience=10, min_improvement=1e-3, reduction_factor=0.5) -> None:
        self.lr = initial_lr
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_loss = np.inf
        self.n_bad_epochs = 0
        self.reduction_factor = reduction_factor

    def update_metric(self, loss):
        if loss < self.best_loss * (1 - self.min_improvement):
            self.best_loss = loss
            self.n_bad_epochs = 0
        else:
            self.n_bad_epochs += 1

        if self.n_bad_epochs > self.patience:
            self.lr *= self.reduction_factor
            self.n_bad_epochs = 0
        return self.lr


@chex.dataclass
class Dataset:
    R: np.ndarray = None
    Z: np.ndarray = None
    n_atoms: np.ndarray = None
    diff: np.ndarray = None
    dist: np.ndarray = None
    edges: np.ndarray = None
    hamiltonian: np.ndarray = None
    core_hamiltonian: np.ndarray = None
    overlap: np.ndarray = None
    density: np.ndarray = None
    energy: np.ndarray = None
    forces: np.ndarray = None
    mask: np.ndarray = None
    ion_mask: np.ndarray = None

    def subset(self, indices):
        return Dataset(
            R=self.R[indices, ...] if self.R is not None else None,
            Z=self.Z[indices, ...] if self.Z is not None else None,
            n_atoms=self.n_atoms[indices, ...] if self.n_atoms is not None else None,
            diff=self.diff[indices, ...] if self.diff is not None else None,
            dist=self.dist[indices, ...] if self.dist is not None else None,
            edges=self.edges[indices, ...] if self.edges is not None else None,
            hamiltonian=self.hamiltonian[indices, ...] if self.hamiltonian is not None else None,
            core_hamiltonian=self.core_hamiltonian[indices, ...] if self.core_hamiltonian is not None else None,
            overlap=self.overlap[indices, ...] if self.overlap is not None else None,
            density=self.density[indices, ...] if self.density is not None else None,
            energy=self.energy[indices, ...] if self.energy is not None else None,
            forces=self.forces[indices, ...] if self.forces is not None else None,
            mask=self.mask[indices, ...] if self.mask is not None else None,
            ion_mask=self.ion_mask[indices, ...] if self.ion_mask is not None else None,
        )

    def shuffle(self, rng):
        idx = jax.random.permutation(rng, self.n_samples)
        return self.subset(idx)

    @property
    def n_samples(self):
        return self.Z.shape[0]


def create_dataset(data, logger=None, shuffle=True, nb_orbitals_per_Z=None):
    max_per_orb_type = {}
    for _, orbs in nb_orbitals_per_Z.items():
        for orb_typ, nb_orb in orbs.items():
            if orb_typ in max_per_orb_type.keys() and max_per_orb_type[orb_typ] < nb_orb:
                max_per_orb_type[orb_typ] = nb_orb
            else:
                max_per_orb_type[orb_typ] = nb_orb

    N_ions_max = get_ions_max(data)
    N_basis = sum(max_per_orb_type.values())
    N_samples = len(data)

    edge_connectivities = []
    for n in range(N_ions_max + 1):
        indices = [(i, j) for i in range(n) for j in range(n) if i != j]
        edge_connectivities.append(np.array(indices, dtype=np.int32))

    full_matrix_shape = [N_samples, N_ions_max, N_ions_max, N_basis, N_basis]
    H_pad = np.zeros(full_matrix_shape)
    H_core_pad = np.zeros(full_matrix_shape)
    S_pad = np.zeros(full_matrix_shape)
    rho_pad = np.zeros(full_matrix_shape)
    Z_pad = np.zeros([N_samples, N_ions_max], dtype=np.int32)
    R_pad = np.zeros([N_samples, N_ions_max, 3], dtype=np.float32)
    diff_pad = np.zeros([N_samples, N_ions_max * (N_ions_max - 1), 3])
    n_atoms_pad = np.zeros([N_samples], dtype=np.int32)
    edge_con = np.ones([N_samples, N_ions_max * (N_ions_max - 1), 2], dtype=np.int32) * N_ions_max + 1
    energies = np.zeros([N_samples], dtype=np.float32)
    forces_pad = np.zeros([N_samples, N_ions_max, 3], dtype=np.float32)
    mask = np.zeros_like(H_pad)
    ion_mask = np.zeros([N_samples, N_ions_max], dtype=np.int32)

    for i, datapoint in enumerate(data):
        if logger is not None and i % 1000 == 0:
            logger.info(f"Processing datapoint {i} of {len(data)}")
        R, Z, H, S, H_core, mo_coeff = datapoint["R"], datapoint["Z"], datapoint["H"], datapoint["S"], datapoint["H_core"], datapoint["mo_coeff"][0]
        n_electrons = int(sum(Z))
        assert n_electrons % 2 == 0, "Assuming close-shell systems; Number of electrons must be even"
        n_occ = n_electrons // 2
        mo_occ = mo_coeff[:, :n_occ]
        rho = 2 * mo_occ @ mo_occ.T
        n_atoms = len(Z)
        n_atoms_pad[i] = n_atoms
        n_edges = n_atoms * (n_atoms - 1)
        R_pad[i, :n_atoms] = R
        Z_pad[i, :n_atoms] = Z
        edge_con[i, :n_edges] = edge_connectivities[n_atoms]
        diff_pad[i, :n_edges] = R_pad[i, edge_con[i, :n_edges, 0], :] - R_pad[i, edge_con[i, :n_edges, 1], :]
        energies[i] = datapoint["E"]
        forces_pad[i, :n_atoms] = datapoint["forces"]
        ion_mask[i, :n_atoms] = 1

        offset_row = 0
        for row, Z_row in enumerate(Z):
            orb_per_row = nb_orbitals_per_Z[Z_row]

            # slice_row_tgt = slice(norb_row)
            norb_row, norb_row_src = 0, 0
            for orb_type_row, nb_orb_per_type_row in orb_per_row.items():
                offset_col = 0

                slice_row_tgt = slice(norb_row, nb_orb_per_type_row + norb_row)
                slice_row_src = slice(norb_row_src + offset_row, norb_row_src + offset_row + nb_orb_per_type_row)

                for col, Z_col in enumerate(Z):
                    orb_per_col = nb_orbitals_per_Z[Z_col]

                    norb_col, norb_col_src = 0, 0
                    for orb_type_col, nb_orb_per_type_col in orb_per_col.items():
                        slice_col_tgt = slice(norb_col, nb_orb_per_type_col + norb_col)
                        slice_col_src = slice(norb_col_src + offset_col, norb_col_src + nb_orb_per_type_col + offset_col)

                        H_pad[i, row, col, slice_row_tgt, slice_col_tgt] = H[slice_row_src, slice_col_src]
                        S_pad[i, row, col, slice_row_tgt, slice_col_tgt] = S[slice_row_src, slice_col_src]
                        H_core_pad[i, row, col, slice_row_tgt, slice_col_tgt] = H_core[slice_row_src, slice_col_src]
                        rho_pad[i, row, col, slice_row_tgt, slice_col_tgt] = rho[slice_row_src, slice_col_src]
                        mask[i, row, col, slice_row_tgt, slice_col_tgt] = 1

                        norb_col += max_per_orb_type[orb_type_col]
                        norb_col_src += nb_orb_per_type_col
                    offset_col += sum(orb_per_col.values())
                norb_row += max_per_orb_type[orb_type_row]
                norb_row_src += nb_orb_per_type_row
            offset_row += sum(orb_per_row.values())


    dist_pad = np.linalg.norm(diff_pad, axis=-1)
    return Dataset(
        R=np.array(R_pad),
        Z=np.array(Z_pad),
        n_atoms=np.array(n_atoms_pad),
        edges=np.array(edge_con, int),
        diff=np.array(diff_pad),
        dist=np.array(dist_pad),
        hamiltonian=np.array(H_pad),
        core_hamiltonian=np.array(H_core_pad),
        overlap=np.array(S_pad),
        density=np.array(rho_pad),
        energy=np.array(energies),
        forces=np.array(forces_pad),
        mask=np.array(mask),
        ion_mask=np.array(ion_mask),
    )

def train_test_split(data: Dataset, trainingset_size, batch_size, shuffle=True, rng=None):
    if trainingset_size > 1:
        trainingset_size = (trainingset_size // batch_size) * batch_size
    else:
        n_batches = int(data.n_samples * trainingset_size / batch_size)
        trainingset_size = n_batches * batch_size
    testset_size = ((data.n_samples - trainingset_size) // batch_size) * batch_size
    assert (trainingset_size > 0) and (testset_size > 0), f"Not enough data to create training and test set: {len(data)}"

    if shuffle:
        rand_idx = jax.random.permutation(rng, data.n_samples)
        train_idx, test_idx = rand_idx[:trainingset_size], rand_idx[-testset_size:]
    else:
        train_idx = np.arange(trainingset_size)
        test_idx = np.arange(trainingset_size, trainingset_size+testset_size)
    training_set = data.subset(train_idx)
    test_set = data.subset(test_idx)
    return training_set, test_set


def unpad_matrices(mask, *matrices):
    """Unpad a matrix by removing zero rows and columns."""
    # Reshape matrix from [N, N, b, b] to [N, b, N, b] and then flatten to []
    mask = jnp.moveaxis(mask, -3, -2).flatten()
    output_size = int(mask.sum())
    n_basis_total = math.isqrt(output_size)
    assert n_basis_total**2 == output_size

    matrices = [jnp.moveaxis(m, -3, -2).flatten() for m in matrices]
    return [m[mask > 0].reshape([n_basis_total, n_basis_total]) for m in matrices]


def build_batches(dataset, batch_size, rng=None, shuffle=True):
    dataset_size = dataset.n_samples
    if shuffle:
        dataset = dataset.shuffle(rng)

    batches = []
    for i in range(dataset_size // batch_size):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        batches.append(dataset.subset(batch_slice))
    return batches


def get_ions_max(data):
    return max([len(mol["Z"]) for mol in data])


def _setup_environment(config: PhisNetTrainingConfiguration):
    root_logger = build_dpe_root_logger(config.logging.basic)

    """ Set random seed """
    rng_seed = np.random.randint(2**31, size=())
    np.random.seed(rng_seed)
    rng_seed = jax.random.PRNGKey(rng_seed)

    params, opt_state = None, None

    return root_logger, rng_seed, config, params, opt_state


def build_model(model_config, batch, rng_seed, irreps_basis):
    model = hk.without_apply_rng(
        hk.transform(
            lambda R_, Z_, edge_ind_: E3PhisNet(
                irreps_basis=irreps_basis,
                n_iterations=model_config.n_iterations,
                L=model_config.L,
                n_channels=model_config.n_channels,
                Z_max=model_config.Z_max,
                n_rbf_features=model_config.n_rbf_features,
                r_max=model_config.r_cutoff,
                r_scale=model_config.r_scale,
                force_overlap_diag_to_one=model_config.force_overlap_diag_to_one,
                predict_overlap=model_config.predict_S,
                predict_hamiltonian=model_config.predict_H,
                predict_core_hamiltonian=model_config.predict_H_core,
                predict_density=model_config.predict_rho,
                predict_energy=model_config.predict_energy,
                predict_forces=model_config.predict_forces,
            )(R_, Z_, edge_ind_)
        )
    )

    if isinstance(batch, Dataset):
       batch = batch.R, batch.Z, batch.edges
    params = model.init(rng_seed, *batch)
    print(get_param_size_summary(params))
    return model, params



def log_params(params, config, epoch, metric, remove_old=False):
    if remove_old:
        fnames = [f for f in os.listdir() if os.path.isfile(f)]
        for fname in fnames:
            match = re.match(r"training_chkpt_epoch_", fname)
            if not match:
                continue
            os.remove(fname)

    data = RunData(config=config, params=jax.tree_util.tree_map(np.array, params))
    fname = os.path.join(".", f"training_chkpt_epoch_{epoch}_metric_{metric}.zip")
    save_run(fname, data)



def train_phisnet(config_file):
    config = PhisNetTrainingConfiguration.load(config_file)
    config.save("full_config.yml")
    root_logger, rng, config, params, opt_state = _setup_environment(config)
    rng, rng_data, rng_model = jax.random.split(rng, 3)

    """ Initialize data/ training & test set """
    root_logger.info(f"Loading data {config.data_path}")
    with open(config.data_path, "rb") as f:
        data = pickle.load(f)

    N_ions_max = get_ions_max(data)
    root_logger.info(f"Maximum number of ions: {N_ions_max}")



    # TODO: remove hardcoding
    all_elements = [1, 4, 5, 6, 7, 8, 9]
    all_atomic_orbitals = _get_all_basis_functions(all_elements, config.basis_set)
    nb_orbitals_per_Z, irreps_basis = _get_orbital_mapping(all_atomic_orbitals, all_elements)
    root_logger.info(f"Number of orbitals per Z: {nb_orbitals_per_Z}")
    full_dataset = create_dataset(data, root_logger, nb_orbitals_per_Z=nb_orbitals_per_Z)
    training_set, test_set = train_test_split(full_dataset, config.trainingset_size, config.batch_size, shuffle=True, rng=rng_data)

    batches = build_batches(training_set, config.batch_size, shuffle=False)
    root_logger.info(f"Number of batches: {len(batches)}")

    """ Initialize model """
    root_logger.info(f"Init PhisNet model...")
    model, params = build_model(config.model, batches[0], rng_model, irreps_basis)
    root_logger.info(f"Nb of params: {hk.data_structures.tree_size(params)}")

    if config.load_checkpoint:
        root_logger.info(f"Loading checkpoint {config.load_checkpoint}")
        checkpoint_data = load_run(config.load_checkpoint, parse_config=False, parse_csv=False, load_pkl=True)
        assert hk.data_structures.tree_size(checkpoint_data.params) == hk.data_structures.tree_size(params), "Number of parameters don't match: checkpoint: {}, model: {}".format(
            hk.data_structures.tree_size(checkpoint_data.params), hk.data_structures.tree_size(params)
        )
        params = checkpoint_data.params

    """ Initialize optimizer """
    if config.lr_schedule_patience > 0:
        optimizer = getattr(optax, config.optimizer.name)
        optimizer = optax.inject_hyperparams(optimizer)(learning_rate=config.optimizer.learning_rate)
        lr_schedule = PatienceLRSchedule(config.optimizer.learning_rate, config.lr_schedule_patience)
    else:
        optimizer = build_optax_optimizer(config.optimizer)
    opt_state = optimizer.init(params)

    """ Initialize training loggers """
    use_wandb_group = False
    exp_idx_in_group = None
    training_loggers: LoggerCollection = initialize_training_loggers(
        config, params, {}, use_wandb_group, exp_idx_in_group
    )
    for l in training_loggers.loggers:
        l.include_epoch = False

    @jax.jit
    def loss_func(params, batch: Dataset):
        _, outputs = model.apply(params, batch.R, batch.Z, batch.edges)
        n_entries = jnp.sum(batch.mask, axis=[-4, -3, -2, -1])
        n_atoms = jnp.sum(batch.ion_mask, axis=-1)

        def masked_matrix_residual(X, X_pred):
            residual = (X - X_pred) * batch.mask
            loss = jnp.sum(residual**2, axis=[-4, -3, -2, -1]) / n_entries
            return jnp.mean(loss) # mean over batch
        
        loss = 0.0
        # Loss terms for S, H, H_core, rho, energy, forces
        if "S" in outputs:
            outputs["loss_S"] = masked_matrix_residual(batch.overlap, outputs["S"])
            loss += outputs["loss_S"] * config.loss_weights.S

        if "H" in outputs:
            outputs["loss_H"] = masked_matrix_residual(batch.hamiltonian, outputs["H"])
            loss += outputs["loss_H"] * config.loss_weights.H

        if "H_core" in outputs:
            outputs["loss_H_core"] = masked_matrix_residual(batch.core_hamiltonian, outputs["H_core"])
            loss += outputs["loss_H_core"] * config.loss_weights.H_core
    
        if "rho" in outputs:
            outputs["loss_rho"] = masked_matrix_residual(batch.density, outputs["rho"])
            loss += outputs["loss_rho"] * config.loss_weights.rho

        if "energy" in outputs:
            residual = batch.energy - outputs["energy"]
            outputs["loss_energy"] = jnp.mean(residual**2)
            loss += outputs["loss_energy"] * config.loss_weights.energy

        if "forces" in outputs:
            residual = (batch.forces - outputs["forces"]) * batch.ion_mask[..., :, None]
            residual = jnp.sum(residual**2, axis=[-1, -2]) / n_atoms
            outputs["loss_forces"] = jnp.mean(residual)
            loss += outputs["loss_forces"] * config.loss_weights.forces

        outputs["loss"] = loss
        return loss, outputs

    @jax.jit
    def step(params, opt_state, batch):
        (loss, aux), g = jax.value_and_grad(loss_func, has_aux=True)(params, batch)
        aux["grad_norm_sqr"] = 0.0
        for g_ in jax.tree_util.tree_leaves(g):
            aux["grad_norm_sqr"] += jnp.sum(g_**2)
        if config.max_grad_norm:
            factor = config.max_grad_norm / jnp.sqrt(aux["grad_norm_sqr"])
            factor = jnp.minimum(factor, 1.0)
            aux["norm_constraint_factor"] = factor
            g = jax.tree_util.tree_map(lambda x: x * factor, g)
        g, opt_state = optimizer.update(g, opt_state)
        params = optax.apply_updates(params, g)
        return loss, aux, params, opt_state

    def evaluate_model(params, dataset: Dataset, diagonalize=True, metrics_prefix=""):
        outputs = dict()
        # Compute predictions and losses per batch
        for batch in build_batches(dataset, config.batch_size, shuffle=False):
            _, aux = loss_func(params, batch)
            for k, v in aux.items():
                if k not in outputs:
                    outputs[k] = []
                outputs[k].append(v)

        # Concatenate all batches and compute errors as mean over batches
        metrics = dict()
        if "atom_embeddings" in outputs:
            del outputs["atom_embeddings"]
        for k,v in outputs.items():
            if k.startswith("loss"):
                metrics[k] = np.mean(v)
            else:
                if isinstance(v[0], jax.Array):
                    outputs[k] = jnp.concatenate(v, axis=0)
                elif isinstance(v[0], e3nn.IrrepsArray):
                    outputs[k] = e3nn.concatenate(v, axis=0)
                else:
                    raise ValueError(f"Unsupported output type: {type(v)}")
        
        if "energy" in outputs:
            metrics["mae_energy"] = np.mean(np.abs(outputs["energy"] - dataset.energy))
        if "forces" in outputs:
            metrics["mae_forces_per_atom"] = np.mean(np.linalg.norm(outputs["forces"] - dataset.forces, axis=-1))
            

        if diagonalize and ("S" in outputs) and ("H" in outputs) and ("rho" not in outputs):
            # Compute mo_coeffs by diagonalization and losses that depend on them
            error_rho, error_mo, mo_cosine_dist = [], [], []
            for i, (S_pad, H_pad, S_ref, H_ref, mask, Z) in enumerate(
                zip(outputs["S"], outputs["H"], dataset.overlap, dataset.hamiltonian, dataset.mask, dataset.Z)
            ):
                S_pred, H_pred, S_ref, H_ref = unpad_matrices(mask, S_pad, H_pad, S_ref, H_ref)
                _, mo_coeff = solve_roothaan_equations(H_pred, S_pred, eps=config.eps_roothaan, eps_mode=config.eps_roothaan_mode)
                _, mo_coeff_ref = solve_roothaan_equations(H_ref, S_ref, eps=config.eps_roothaan, eps_mode=config.eps_roothaan_mode)

                n_el = int(np.sum(Z))
                n_occ = n_el // 2
                mo_coeff = align_orbitals(mo_coeff, mo_coeff_ref, n_occ)

                rho = mo_coeff[:, :n_occ] @ mo_coeff[:, :n_occ].T
                rho_ref = mo_coeff_ref[:, :n_occ] @ mo_coeff_ref[:, :n_occ].T
                error_rho.append(np.nanmean((rho - rho_ref) ** 2))
                error_mo.append(np.nanmean(mo_coeff - mo_coeff_ref) ** 2)
                mo_cosine_dist.append(get_cosine_dist(mo_coeff[:, :n_occ], mo_coeff_ref[:, :n_occ]))
            metrics["error_rho"] = np.nanmean(error_rho)
            metrics["error_mo"] = np.nanmean(error_mo)
            metrics["mo_occ_cosine_dist"] = np.nanmean(mo_cosine_dist)

        if metrics_prefix:
            metrics = {f"{metrics_prefix}_{k}": v for k, v in metrics.items()}
        return metrics, outputs

    root_logger.info("Starting optimization....")
    ind_step = 0
    prev_best_loss = np.inf
    for ind_epoch in range(config.n_epochs):
        are_test_metrics_new = False
        need_checkpoint_for_accuracy = False
        metrics = dict(step=ind_step, epoch=ind_epoch, lr=lr_schedule.lr if config.lr_schedule_patience > 0 else None)
        if (config.validate_full_every_n_epochs > 0) and (ind_epoch % config.validate_full_every_n_epochs == 0) and (ind_epoch > 0):
            metrics_train, _ = evaluate_model(params, training_set, metrics_prefix="train")
            metrics_test, _ = evaluate_model(params, test_set, metrics_prefix="test")
            metrics.update(metrics_train)
            metrics.update(metrics_test)
            are_test_metrics_new = True
        elif (config.validate_small_every_n_epochs > 0) and (ind_epoch % config.validate_small_every_n_epochs == 0) and (ind_epoch > 0):
            metrics_test, _ = evaluate_model(params, test_set, metrics_prefix="test", diagonalize=False)
            metrics.update(metrics_test)
            are_test_metrics_new = True

        if are_test_metrics_new:
            if config.lr_schedule_patience > 0:
                # Update the learning rate using the patience scheduler
                opt_state.hyperparams["learning_rate"] = lr_schedule.update_metric(metrics_test["test_loss"])
            if metrics_test.get(config.checkpoint.checkpoint_metric, np.inf) < prev_best_loss:
                need_checkpoint_for_accuracy = True
                prev_best_loss = metrics_test[config.checkpoint.checkpoint_metric]
        need_checkpoint_for_epoch = (ind_epoch > 0) and (config.checkpoint.every_n_epochs > 0) and (ind_epoch % config.checkpoint.every_n_epochs == 0)

        if need_checkpoint_for_accuracy or need_checkpoint_for_epoch:
            root_logger.info(
                f"Logging params for metric {config.checkpoint.checkpoint_metric}: {metrics_test[config.checkpoint.checkpoint_metric]}")
            log_params(params, 
                       config, 
                       ind_epoch,
                        metrics_test[config.checkpoint.checkpoint_metric], 
                        remove_old=need_checkpoint_for_accuracy)

        training_loggers.log_metrics(
            metrics,
            epoch=ind_step,
            force_log=True,
        )
        if config.lr_schedule_patience > 0 and lr_schedule.lr < 1e-6:
            root_logger.info(f"Learning rate too small (lr={lr_schedule.lr}), stopping training")
            break

        rng, key = jax.random.split(rng)
        # Actual training loop, running over batches
        for ind_batch, batch in enumerate(build_batches(training_set, config.batch_size, key, shuffle=True)):
            loss, aux, params, opt_state = step(params, opt_state, batch)
            metrics = {k:v for k,v in aux.items() if k.startswith("loss")}
            metrics["norm_constraint_factor"] = aux.get("norm_constraint_factor")
            metrics["grad_norm"] = np.sqrt(aux["grad_norm_sqr"])
            metrics["step"] = ind_step
            training_loggers.log_metrics(metrics, epoch=ind_step)
            if np.isnan(loss):
                raise ValueError("Loss is NaN. Aborting.")
            ind_step += 1


    # Switch to log checkpoint
    log_params(params, config, config.n_epochs, "final_params")

    # data = RunData(config=config, params=jax.tree_util.tree_map(np.array, params))
    # fname = os.path.join(".", f"training_chkpt_final.zip")
    # save_run(fname, data)

    # Evaluate model and store predictions
    for data, name in zip([training_set, test_set], ["train", "test"]):
        root_logger.info(f"Evaluating model on {name}...")
        metrics, outputs = evaluate_model(params, data, metrics_prefix=name)
        np.savez_compressed(
            f"predictions_{name}.npz",
            **outputs,
            **metrics,
            mask=data.mask,
            S_ref=data.overlap,
            H_ref=data.hamiltonian,
            H_core_ref=data.core_hamiltonian,
            rho_ref=data.density,
            energy_ref=data.energy,
            forces_ref=data.forces,
            R=data.R,
            Z=data.Z,
            n_atoms=data.n_atoms,
        )

    training_loggers.on_run_end()


if __name__ == "__main__":
    train_phisnet(sys.argv[1])
