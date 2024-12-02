import h5py
from deeperwin.model.orbitals.generalized_atomic_orbitals import GAOExponents
from deeperwin.orbitals import get_atomic_orbital_descriptors, OrbitalParamsHF, AtomicOrbital, _get_all_basis_functions
from deeperwin.configuration import MLPConfig
import numpy as np
import haiku as hk
import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from deeperwin.checkpoints import RunData, save_run


def load_orbital_params(HF_data: h5py.Group):
    ao_group = HF_data["atomic_orbitals"]
    aos = [_build_atomic_orbital(ao_group[str(i)]) for i in range(len(ao_group))]
    return OrbitalParamsHF(atomic_orbitals=aos, mo_coeff=HF_data["mo_coeff"][...])


def _build_atomic_orbital(ao_data: h5py.Group):
    kwargs = {k: v[...] for k, v in ao_data.items()}
    return AtomicOrbital(**kwargs)


def load_dataset(fname, elements, basis_set, n_geoms=None, print_progress=True):
    all_atomic_orbitals = _get_all_basis_functions(elements, basis_set)
    ao_mapping = {(int(ao.Z), int(ao.idx_basis)): i for i, ao in enumerate(all_atomic_orbitals)}

    all_features = []
    all_exponents = []
    all_prefacs = []
    with h5py.File(fname, "r") as f:
        geom_ids = list(f.keys())
        if n_geoms:
            geom_ids = np.random.choice(geom_ids, n_geoms, replace=False)
        for i, geom_id in enumerate(geom_ids):
            geom = f[geom_id]
            if f"HF/{basis_set}/exponents" not in geom:
                continue
            if print_progress and (i % 20 == 0):
                print(f"Loading geom {i:6d}")
            HF_data = geom[f"HF/{basis_set}"]
            orb_params = load_orbital_params(HF_data)
            n_up, n_dn = int(HF_data["n_up"][...]), int(HF_data["n_dn"][...])
            n_orb = max(n_up, n_dn)
            orb_params.mo_coeff = orb_params.mo_coeff[:, :, : max(n_up, n_dn)]
            features = get_atomic_orbital_descriptors(orb_params, elements, None, ao_mapping)
            all_features.append(np.stack(features, 1))
            all_exponents.append(HF_data["exponents"][:, :, :n_orb])
            all_prefacs.append(HF_data["prefacs"][:, :, :n_orb])
    return all_features, all_exponents, all_prefacs


def flatten_dataset(dataset, spin=None):
    if spin is None:
        spin_slice = slice(None)
    else:
        spin_slice = slice(spin, spin + 1)
    return np.concatenate([x[:, spin_slice].reshape((-1, *x.shape[3:])) for x in dataset], axis=0)


def make_batches(x, y, batch_size):
    n_samples = len(x)
    n_batches = n_samples // batch_size
    assert n_samples % batch_size == 0, "Dataset size is not evenly divisible into batches"
    ind_shuffle = np.random.permutation(len(x))
    x_batches = x[ind_shuffle].reshape((n_batches, batch_size, *x.shape[1:]))
    y_batches = y[ind_shuffle].reshape((n_batches, batch_size, *y.shape[1:]))
    return x_batches, y_batches


def split_train_test(x, y, batch_size, train_ratio=0.8):
    ind_shuffle = np.random.permutation(len(x))
    n_train = (int(len(x) * train_ratio) // batch_size) * batch_size
    ind_train, ind_test = np.split(ind_shuffle, [n_train])
    return x[ind_train], y[ind_train], x[ind_test], y[ind_test]


def transform_params_for_export(params, n_dets, w_noise=1e-2, b_noise=1e-2):
    n_layers = len(params.keys())
    params_out = dict()
    for l in range(n_layers):
        src_name = f"gao_exponents/mlp/linear_{l}"
        for spin in range(2):
            target_name = f"wf/orbitals/generalized_atomic_orbitals/exponent_{spin}/mlp/linear_{l}"
            if l != (n_layers - 1):
                params_out[target_name] = jax.tree_util.tree_map(lambda x: x, params[src_name])
            else:
                p = jax.tree_util.tree_map(lambda x: x, params[src_name])
                p["w"] = p["w"].reshape([-1, 2, 1, 2])  # [features x same/diff x det x exp/prefac]
                p["w"] = jnp.tile(p["w"], [n_dets, 1])  # use the same parameters across determinants
                p["w"] = p["w"].reshape([-1, 4 * n_dets])

                p["b"] = p["b"].reshape([2, 1, 2])  # [features x same/diff x det x exp/prefac]
                p["b"] = jnp.tile(p["b"], [n_dets, 1])  # use the same parameters across determinants
                p["b"] = p["b"].reshape([4 * n_dets])

                p["w"] *= np.random.normal(loc=1.0, scale=w_noise, size=p["w"].shape)
                p["b"] *= np.random.normal(loc=1.0, scale=b_noise, size=p["b"].shape)
                params_out[target_name] = p
    return params_out


if __name__ == "__main__":
    fname = "/home/mscherbela/runs/datasets/QM7HEAT_36el/QM7HEAT_36el_2000geom.hdf5"
    basis_set = "6-311G**"
    elements = [1, 6, 7, 8]
    learning_rate = lambda t: 5e-4 / (1 + t / 100_000)
    n_epochs = 100
    batch_size = 32
    prefac_ref_offdiag = 0.1
    exp_ref_offdiag = 2.0
    n_geoms = None
    ema_factor = 0.999
    reload_dataset = False
    n_dets_export = 4
    width = 64
    depth = 2
    activation = "tanh"
    output_fname = (
        f"/home/mscherbela/runs/datasets/QM7HEAT_36el/GAO_env_params_{n_dets_export}fd_{width}x{depth}_{activation}.zip"
    )

    if reload_dataset or "features" not in locals():
        print("Loading dataset...")
        features, exponents, prefacs = load_dataset(fname, elements, basis_set, n_geoms)
        n_geoms = len(features)
        features, exponents, prefacs = [flatten_dataset(x, spin=0) for x in [features, exponents, prefacs]]
        targets = np.stack([exponents, prefacs], axis=-1)
        features_train, targets_train, features_test, targets_test = split_train_test(features, targets, batch_size)

    optimizer = optax.adam(learning_rate)
    model = hk.without_apply_rng(
        hk.transform(
            lambda c: GAOExponents(
                width=width,
                depth=depth,
                determinant_schema="full_det",
                n_dets=1,
                symmetrize=True,
                mlp_config=MLPConfig(activation="tanh"),
            )(c)
        )
    )
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, features[:batch_size, ...])
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_func(params, batch):
        x, y = batch
        exp_ref_diag, prefac_ref_diag = y[..., 0], y[..., 1]
        prediction = model.apply(params, x)

        # Only use main-diagonal, single determinant for now
        exp_pred_diag = prediction[0][..., 0, 0]
        exp_pred_offdiag = prediction[0][..., 1, 0]
        prefac_pred_diag = prediction[1][..., 0, 0]
        prefac_pred_offdiag = prediction[1][..., 1, 0]

        eps = 1.0
        res_prefac_diag = prefac_pred_diag - prefac_ref_diag
        res_prefac_offdiag = prefac_pred_offdiag - prefac_ref_offdiag
        res_exp_diag = exp_pred_diag - exp_ref_diag
        res_exp_offdiag = exp_pred_offdiag - exp_ref_offdiag

        # L2 penalty for prefac; weighted L2 penalty for exponents (don't care about exponents on 0-weighted envelopes)
        loss = 0.0
        loss += jnp.mean(res_prefac_diag**2, axis=0)
        loss += jnp.mean(res_prefac_offdiag**2, axis=0)
        loss += jnp.mean((prefac_ref_diag + eps) ** 2 * res_exp_diag**2, axis=0)
        loss += jnp.mean((prefac_ref_offdiag + eps) ** 2 * res_exp_offdiag**2, axis=0)
        # loss += jnp.mean(res_exp_diag ** 2, axis=0)
        # loss += jnp.mean(res_exp_offdiag ** 2, axis=0)
        return loss

    @jax.jit
    def step(params, opt_state, batch):
        loss, g = jax.value_and_grad(loss_func)(params, batch)
        update, opt_state = optimizer.update(g, opt_state)
        params = optax.apply_updates(params, update)
        return loss, params, opt_state

    @jax.jit
    def ema(params_old, params_new):
        return jax.tree_map(lambda o, n: ema_factor * o + (1 - ema_factor) * n, params_old, params_new)

    loss_values = []
    n_epoch_values = []
    loss_test = []
    loss_train = []
    params_ema = params
    for ind_epoch in range(n_epochs):
        for batch in zip(*make_batches(features_train, targets_train, batch_size)):
            loss, params, opt_state = step(params, opt_state, batch)
            params_ema = ema(params_ema, params)
            loss_values.append(loss)
        if (ind_epoch % 5) == 0 or (ind_epoch == n_epochs - 1):
            n_epoch_values.append(ind_epoch)
            loss_train.append(loss_func(params, (features_train, targets_train)))
            loss_test.append(loss_func(params, (features_test, targets_test)))
            print(f"epoch={ind_epoch}, loss train={loss_train[-1]:6.4f}, loss test={loss_test[-1]:6.4f}")
    n_step_values = np.arange(len(loss_values))

    params_out = transform_params_for_export(params_ema, n_dets_export)
    save_run(output_fname, RunData(params=params_out))

    # %%
    plt.close("all")
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    axes[0][0].semilogy(n_step_values / 1000, loss_values)
    axes[0][0].set_xlabel("Step / k")
    axes[0][0].set_ylabel("Batch loss")

    axes[1][0].semilogy(n_epoch_values, loss_train, label="Train loss")
    axes[1][0].semilogy(n_epoch_values, loss_test, label="Test loss")
    axes[1][0].set_xlabel("Epoch")
    axes[1][0].set_ylabel("Full loss")
    axes[1][0].legend()
    axes[1][0].set_ylim([None, 0.2])

    for dataset, scatter_axes, label, color in zip(
        [(features_train, targets_train), (features_test, targets_test)],
        [axes[:, 1], axes[:, 2]],
        ["Train", "Test"],
        ["C0", "C1"],
    ):
        exponents_pred, prefacs_pred = model.apply(params_ema, dataset[0])
        loss_ema = loss_func(params_ema, dataset)
        print(f"EMA loss {label}: {loss_ema: .3f}")
        exponents_ref, prefacs_ref = np.split(dataset[1], 2, axis=-1)
        exp_range = [np.min(exponents_ref), np.max(exponents_ref)]
        prefac_range = [np.min(prefacs_ref), np.max(prefacs_ref)]
        scatter_axes[0].plot(exp_range, exp_range, color="k", ls="--")
        scatter_axes[0].scatter(
            exponents_ref.flatten(),
            exponents_pred[..., 0, 0],
            alpha=0.4,
            s=np.sqrt(prefacs_ref.flatten() * 500),
            color=color,
            edgecolor="none",
        )
        scatter_axes[0].set_xlabel("exp ref")
        scatter_axes[0].set_ylabel("exp pred")
        scatter_axes[0].set_title(f"Exponents {label}")
        scatter_axes[1].plot(prefac_range, prefac_range, color="k", ls="--")
        scatter_axes[1].scatter(
            prefacs_ref.flatten(), prefacs_pred[..., 0, 0], alpha=0.4, color=color, edgecolor="none"
        )
        scatter_axes[1].set_xlabel("$\pi$ ref")
        scatter_axes[1].set_ylabel("$\pi$ pred")
        scatter_axes[1].set_title(f"Prefactors {label}")

        fig.suptitle(f"Pretraining of envelopes on {n_geoms} geometries; {basis_set}", fontsize=16)
        fig.tight_layout()
        fig_fname = f"/home/mscherbela/ucloud/results/envelope_pretraining_{n_geoms}geoms_{basis_set}.png"
        fig.savefig(fig_fname, dpi=400, bbox_inches="tight")
        # fig.savefig(fig_fname.replace(".png", ".pdf"), bbox_inches='tight')
