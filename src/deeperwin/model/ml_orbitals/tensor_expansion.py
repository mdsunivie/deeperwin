import numpy as np
#%%
import e3nn_jax as e3nn

irreps1 = e3nn.Irreps("1o")
irreps2 = e3nn.Irreps("1o")
n_samples = (irreps1.dim * irreps2.dim)

x = np.random.normal(size=[n_samples, irreps1.dim])
y = np.random.normal(size=[n_samples, irreps2.dim])

M = np.einsum("...i,...j->...ij", x, y)
tp = e3nn.tensor_product(e3nn.IrrepsArray(irreps1, x), e3nn.IrrepsArray(irreps2, y)).array


# U, res, _, s = np.linalg.lstsq(tp, M.reshape(n_samples, -1), rcond=None)
U = np.linalg.solve(tp, M.reshape(n_samples, -1))
#print("Max. residual:", res.max())
np.set_printoptions(precision=3, suppress=True, linewidth=200)
print(U)

cg = e3nn.clebsch_gordan(1,1,2)

# %%

# %%
