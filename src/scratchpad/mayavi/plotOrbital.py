from itertools import chain
import jax
import jax.numpy as jnp
import numpy as np
from utils import load_from_file, get_el_ion_distance_matrix, get_distance_matrix
from orbitals import get_baseline_solution, get_molecule, eval_atomic_orbitals
from model import build_backflow_factor, build_simple_schnet, get_rbf_features
from mayavi.mlab import contour3d, volume_slice, points3d
from mayavi import mlab
import pathlib

# directory = '/users/mscherbela/runs/jaxtest/viz/LiH_1_f_f_10000_2000_100/'
directory = '/users/mscherbela/runs/jaxtest/viz/LiH_1_f_f_20000_2000_100/'
data = load_from_file(directory + '../LiH_1_f_f_10000_2000/results.bz2')
config = data['config']
params = data['params']
chkpt_data = [load_from_file(f) for f in pathlib.Path(directory).glob('chkpt*.bz2')]
for c in chkpt_data:
    del c['mcmc'] # free some memory
chkpt_data.sort(key=lambda x: x['n_epoch'])

R, Z, n_electrons, n_up, el_ion_mapping, n_cas_orbitals, n_cas_electrons = get_molecule(config['molecule'])
baseline_solution, (E_hf, E_cas) = get_baseline_solution(R, Z, **config)
atomic_orbitals, cusp_params, mo_coeff, ind_orb, ci_weights = baseline_solution

_, calc_embedding, _ = build_simple_schnet(config)
_, calc_bf_factor, _ = build_backflow_factor(config)

@jax.jit
def calculate_backflow_distortion(r, params):
    diff_el_el, dist_el_el = get_distance_matrix(r)
    _, dist_el_ion = get_el_ion_distance_matrix(r, R)
    features_el_el = get_rbf_features(dist_el_el, config["n_rbf_features"])
    features_el_ion = get_rbf_features(dist_el_ion, config["n_rbf_features"])
    embeddings = calc_embedding(features_el_el, features_el_ion, Z, params['embed'])
    bf_up, bf_dn = calc_bf_factor(embeddings, params['bf_fac'])
    return bf_up, bf_dn

if config['molecule'] == 'LiH':
    r0 = jnp.stack(np.mgrid[-2:5:60j, -1:1:30j, -1:1:30j], axis=-1)
elif config['molecule'] == 'H2':
    r0 = jnp.stack(np.mgrid[-0.5:1.9:60j, -1:1:30j, -1:1:30j], axis=-1)
else:
    raise ValueError("unknown molecule")

r_others = np.array(data['mcmc'][0][0][1:])
# r_others = np.array([[1.0, 0, 0]])
r = jnp.concatenate([r0[..., np.newaxis, :], jnp.tile(r_others, r0.shape[:3]+(1,1))], axis=-2)

el_ion_diff, el_ion_dist = get_el_ion_distance_matrix(r, R)
ao_matrix = eval_atomic_orbitals(el_ion_diff, el_ion_dist, atomic_orbitals, cusp_params)
mo_matrix_up = ao_matrix[..., jnp.newaxis, :n_up, :] @ mo_coeff[0]
mo_matrix_dn = ao_matrix[..., jnp.newaxis, n_up:, :] @ mo_coeff[1]
bf_up, bf_dn = calculate_backflow_distortion(r, params)

orbital_index = 1
orbital_hf = mo_matrix_up[..., 0, 0, orbital_index].to_py() # first determinant, first electron, nth orbital
orbital_dpe = orbital_hf * bf_up[..., 0, 0, orbital_index].to_py()

#%%
fig = mlab.figure(size=(1300, 900))
fig.scene.movie_maker.record = True
points3d(R[:,0].to_py(), R[:,1].to_py(), R[:,2].to_py(), Z.to_py(), resolution=16, scale_factor=0.3, color=(0.2,0.2,0.2))
points3d(r_others[:,0], r_others[:,1], r_others[:,2], resolution=16, scale_factor=0.1, color=(0,0,1))

# colormap = dict(colormap='RdBu', vmin=-np.abs(orbital_hf).max(), vmax=np.abs(orbital_hf).max())
colormap = {}
handle_slice = volume_slice(r0[...,0], r0[...,1], r0[...,2], orbital_hf, slice_index=r0.shape[2]//2, opacity=0.8, plane_orientation='z_axes', **colormap)
handle_contour = contour3d(r0[...,0], r0[...,1], r0[...,2], orbital_hf, contours=16, opacity=0.3, **colormap)

fig.scene.camera.position = [5.86694104558851, -7.009322394396042, 4.515628538526653]
fig.scene.camera.focal_point = [2.386696423536285, 0.07359423716237518, -0.28856921334652186]
fig.scene.camera.view_angle = 30.0
fig.scene.camera.view_up = [-0.21886899761883374, 0.4717864257468988, 0.854115876425613]
fig.scene.camera.clipping_range = [4.185647138587369, 15.622298163970612]
fig.scene.camera.compute_view_plane_normal()
# mlab.show()

def triangle_linspace(n_steps):
    x_up = np.linspace(0, 1.0, (n_steps+1) // 2)
    x_dn = x_up[:-1][::-1]
    return np.concatenate([x_up, x_dn])

n_animation_steps = 141
exaggeration_factor = 1.0
@mlab.animate(delay=20, ui=True)
def build_animation():
    for scaling in triangle_linspace(n_animation_steps)*exaggeration_factor:
        handle_contour.mlab_source.scalars = orbital_hf * (1 + scaling*(bf_up[..., 0, 0, orbital_index].to_py()-1))
        yield
    # for chkpt in chkpt_data:
    #     print(f"Epoch {chkpt['n_epoch']}")
    #     bf_up, bf_dn = calculate_backflow_distortion(r, chkpt['params'])
    #     scaling = chkpt['n_epoch'] / chkpt_data[-1]['n_epoch']
    #     new_values = orbital_hf * ((bf_up[..., 0, 0, orbital_index].to_py() - 1) * scaling + 1)
    #     handle_contour.mlab_source.scalars = new_values
    #     handle_slice.mlab_source.scalars = new_values
    #     yield

build_animation()
mlab.show()


