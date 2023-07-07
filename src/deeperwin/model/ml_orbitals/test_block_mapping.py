#%%
from deeperwin.model.ml_orbitals.ml_orbitals import E3PhisNetMatrixBlock

if __name__ == "__main__":
    irreps_row = "2x0e+1x1o"
    irreps_col = "2x0e+1x1o"

    irreps_in = e3nn.tensor_product(irreps_row, irreps_col)
    rng, rng1, rng2 = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jax.random.normal(rng1, [irreps_in.dim])
    x = e3nn.IrrepsArray(irreps_in, x)

    # Filter out the 1e
    factor = np.ones(x.irreps.num_irreps)
    factor[9] = 0
    x *= factor

    model = hk.without_apply_rng(
        hk.transform(lambda x: E3PhisNetMatrixBlock(irreps_row, irreps_col)(x))
    )
    params = model.init(rng, x)
    block = model.apply(params, x)

    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print(block)
    print("done")
# %%
