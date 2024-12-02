from deeperwin.orbitals import eval_gaussian_orbital, OrbitalParamsHF, AtomicOrbital
import numpy as np
import jax.numpy as jnp
import functools
import jax
import h5py
import multiprocessing as mp
import argparse


def get_random_points_on_unit_sphere(n):
    r = np.random.normal(size=[n, 3])
    r = r / np.linalg.norm(r, axis=-1, keepdims=True)
    return r


def fit_exponents_from_radial_integral(r, int_phi_sqr, x=2.0):
    integral = 1 - jnp.exp(-2 * x) * ((x + 1) ** 2 + x**2)
    int_phi_sqr_normalized = int_phi_sqr / jnp.max(int_phi_sqr, axis=0, keepdims=True)  # [radius x orbitals]
    r_crit = jnp.interp(integral, int_phi_sqr_normalized, r)
    exponent = x / r_crit
    prefac = jnp.sqrt(exponent**3 * int_phi_sqr[-1] / np.pi)
    return exponent, prefac


@jax.jit
def get_radial_cdf(dr, phi):
    # phi: [batch x radial x angular]
    phi_sqr = jnp.mean(phi**2, axis=-1)
    phi_interval = jnp.concatenate([phi_sqr[..., :1], (phi_sqr[..., 1:] + phi_sqr[..., :-1]) * 0.5], axis=-1)
    return np.cumsum(dr * phi_interval, axis=-1)


def get_integral_weights(r):
    r_lower = np.concatenate([[0], r[:-1]])
    r_upper = r
    dr = 4 * np.pi * (r_upper**3 - r_lower**3) / 3
    return dr


@functools.partial(jax.jit, static_argnums=(3,))
def evaluate_orbitals_per_atom(orbital_params, r, norm_r, n_ions):
    aos = jnp.stack([eval_gaussian_orbital(r, norm_r, a) for a in orbital_params.atomic_orbitals], axis=-1)
    idx_atom = jnp.array([a.idx_atom for a in orbital_params.atomic_orbitals], int)
    ao_mask = jnp.where(idx_atom[None, :] == np.arange(n_ions)[:, None], 1.0, 0.0)
    # r: radial, a: angular, b: basis_func (ao), I: Ion, j: orbital
    phi = jnp.einsum("...rab,Ib,sbj->...Isjra", aos, ao_mask, orbital_params.mo_coeff)
    return phi


def build_orbital_params(HF_data: h5py.Group):
    ao_group = HF_data["atomic_orbitals"]
    aos = [_build_atomic_orbital(ao_group[str(i)]) for i in range(len(ao_group))]
    return OrbitalParamsHF(atomic_orbitals=aos, mo_coeff=HF_data["mo_coeff"][...])


def _build_atomic_orbital(ao_data: h5py.Group):
    kwargs = {k: v[...] for k, v in ao_data.items()}
    return AtomicOrbital(**kwargs)


def process_result(result, f):
    (exponents, prefacs), (ind, calc_name) = result
    print(f"Saving {ind+1:6d}: {calc_name}")
    f.create_dataset(calc_name + "/exponents", data=exponents)
    f.create_dataset(calc_name + "/prefacs", data=prefacs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, help="Path to HDF5 file with geometry dataset")
    parser.add_argument(
        "--n-proc", type=int, default=1, help="Number of processes to spawn for calculations in parallel"
    )
    parser.add_argument("--n-calcs", type=int, default=0, help="Number of geometries to process")
    parser.add_argument("--n-angular", type=int, default=1000)
    parser.add_argument("--n-radial", type=int, default=100)
    parser.add_argument("--r-min", type=float, default=1e-2)
    parser.add_argument("--r-max", type=float, default=5.0)
    args = parser.parse_args()

    for k, v in args._get_kwargs():
        print(f"{k} = {v}")

    radius_values = np.geomspace(args.r_min, args.r_max, args.n_radial)
    dr = get_integral_weights(radius_values)
    r_sphere = get_random_points_on_unit_sphere(args.n_angular)
    r_grid = r_sphere[None, :, :] * radius_values[:, None, None]
    norm_grid = np.linalg.norm(r_grid, axis=-1)
    fit_exp = jax.jit(jax.vmap(jax.vmap(jax.vmap(lambda y: fit_exponents_from_radial_integral(radius_values, y, 2.0)))))

    def compute_envelope_parameters(orbital_params, n_ions, metadata):
        phi = evaluate_orbitals_per_atom(orbital_params, r_grid, norm_grid, n_ions)
        radial_cdf = get_radial_cdf(dr, phi)
        return fit_exp(radial_cdf), metadata

    with h5py.File(args.fname, "a") as f:
        arguments = []
        for ind_geom, (geom_name, geom) in enumerate(f.items()):
            if "HF" not in geom:
                continue
            Z = geom["atNUM"][...]
            n_ions = len(Z)
            for basis_set, HF_data in geom["HF"].items():
                calc_name = f"{geom_name}/HF/{basis_set}"
                if calc_name + "/exponents" in f:
                    print(f"Skipping: {calc_name}")
                    continue
                print(f"Loading: {calc_name}")
                orbital_params = build_orbital_params(HF_data)
                arguments.append((orbital_params, n_ions, (len(arguments), calc_name)))
            if args.n_calcs and len(arguments) >= args.n_calcs:
                break

        print(f"Total nr of calculations: {len(arguments)}")
        with mp.Pool(args.n_proc) as pool:
            for arg in arguments:
                pool.apply_async(compute_envelope_parameters, arg, callback=lambda res: process_result(res, f))

            pool.close()
            pool.join()

    print("Finished calculation of exponents")
