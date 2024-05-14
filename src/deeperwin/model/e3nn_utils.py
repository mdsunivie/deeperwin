from math import sqrt
from typing import Callable, List, NamedTuple, Optional, Tuple, Union
import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irreps, IrrepsArray, config
from e3nn_jax._src.utils import sum_tensors
from jax import numpy as jnp
from deeperwin.model.mlp import get_activation, get_rbf_features, MLP
from deeperwin.configuration import MLPConfig


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple
    path_weight: float
    weight_std: float


class FunctionalLinear:
    irreps_in: Irreps
    irreps_out: Irreps
    instructions: List[Instruction]
    output_mask: jnp.ndarray

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        instructions: Optional[List[Tuple[int, int]]] = None,
        biases: Optional[Union[List[bool], bool]] = None,
        path_normalization: Union[str, float] = None,
        gradient_normalization: Union[str, float] = None,
    ):
        if path_normalization is None:
            path_normalization = config("path_normalization")
        if isinstance(path_normalization, str):
            path_normalization = {"element": 0.0, "path": 1.0}[path_normalization]

        if gradient_normalization is None:
            gradient_normalization = config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[gradient_normalization]

        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)

        if instructions is None:
            # By default, make all possible connections
            instructions = [
                (i_in, i_out)
                for i_in, (_, ir_in) in enumerate(irreps_in)
                for i_out, (_, ir_out) in enumerate(irreps_out)
                if ir_in == ir_out
            ]

        instructions = [
            Instruction(
                i_in=i_in,
                i_out=i_out,
                path_shape=(irreps_in[i_in].mul, irreps_out[i_out].mul),
                path_weight=1,
                weight_std=1,
            )
            for i_in, i_out in instructions
        ]

        def alpha(this):
            x = irreps_in[this.i_in].mul ** path_normalization * sum(
                irreps_in[other.i_in].mul ** (1.0 - path_normalization)
                for other in instructions
                if other.i_out == this.i_out
            )
            return 1 / x if x > 0 else 1.0

        instructions = [
            Instruction(
                i_in=ins.i_in,
                i_out=ins.i_out,
                path_shape=ins.path_shape,
                path_weight=sqrt(alpha(ins)) ** gradient_normalization,
                weight_std=sqrt(alpha(ins)) ** (1.0 - gradient_normalization),
            )
            for ins in instructions
        ]

        if biases is None:
            biases = len(irreps_out) * (False,)
        if isinstance(biases, bool):
            biases = [biases and ir.is_scalar() for _, ir in irreps_out]

        assert len(biases) == len(irreps_out)
        assert all(ir.is_scalar() or (not b) for b, (_, ir) in zip(biases, irreps_out))

        instructions += [
            Instruction(i_in=-1, i_out=i_out, path_shape=(mul_ir.dim,), path_weight=1.0, weight_std=0.0)
            for i_out, (bias, mul_ir) in enumerate(zip(biases, irreps_out))
            if bias
        ]

        with jax.ensure_compile_time_eval():
            if irreps_out.dim > 0:
                output_mask = jnp.concatenate(
                    [
                        jnp.ones(mul_ir.dim)
                        if any((ins.i_out == i_out) and (0 not in ins.path_shape) for ins in instructions)
                        else jnp.zeros(mul_ir.dim)
                        for i_out, mul_ir in enumerate(irreps_out)
                    ]
                )
            else:
                output_mask = jnp.ones(0)

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.instructions = instructions
        self.output_mask = output_mask

    def aggregate_paths(self, paths, output_shape) -> IrrepsArray:
        output = [
            sum_tensors(
                [out for ins, out in zip(self.instructions, paths) if ins.i_out == i_out],
                shape=output_shape
                + (
                    mul_ir_out.mul,
                    mul_ir_out.ir.dim,
                ),
                empty_return_none=True,
            )
            for i_out, mul_ir_out in enumerate(self.irreps_out)
        ]
        return IrrepsArray.from_list(self.irreps_out, output, output_shape)

    def __call__(self, ws: List[jnp.ndarray], input: IrrepsArray) -> IrrepsArray:
        input = input._convert(self.irreps_in)
        if input.ndim != 1:
            raise ValueError(f"FunctionalLinear does not support broadcasting, input shape is {input.shape}")

        paths = []
        for ins, w in zip(self.instructions, ws):
            if ins.i_in == -1:
                y = ins.path_weight * w
            else:
                if input.list[ins.i_in] is None:
                    y = None
                else:
                    input_transpose = input.list[ins.i_in].transpose()
                    y_intermed = jnp.dot(
                        input_transpose, w
                    )  # register_repeated_dense(jnp.dot(input_transpose, w), input_transpose, w, None)
                    # y_intermed = jnp.einsum("uw,ui->wi", w, input.list[ins.i_in])
                    y = ins.path_weight * y_intermed.transpose()

            paths.append(y)
        # paths = [
        #     ins.path_weight * w
        #     if ins.i_in == -1
        #     else (None if input.list[ins.i_in] is None else ins.path_weight * jnp.einsum("uw,ui->wi", w, input.list[ins.i_in]))
        #     for ins, w in zip(self.instructions, ws)
        # ]
        return self.aggregate_paths(paths, input.shape[:-1])

    def matrix(self, ws: List[jnp.ndarray]) -> jnp.ndarray:
        r"""Compute the matrix representation of the linear operator.

        Args:
            ws: List of weights.

        Returns:
            The matrix representation of the linear operator. The matrix is shape ``(irreps_in.dim, irreps_out.dim)``.
        """
        output = jnp.zeros((self.irreps_in.dim, self.irreps_out.dim))
        for ins, w in zip(self.instructions, ws):
            assert ins.i_in != -1
            mul_in, ir_in = self.irreps_in[ins.i_in]
            mul_out, ir_out = self.irreps_out[ins.i_out]
            output = output.at[self.irreps_in.slices()[ins.i_in], self.irreps_out.slices()[ins.i_out]].add(
                ins.path_weight
                * jnp.einsum("uw,ij->uiwj", w, jnp.eye(ir_in.dim)).reshape((mul_in * ir_in.dim, mul_out * ir_out.dim))
            )
        return output


class Linear(hk.Module):
    r"""Equivariant Linear Haiku Module.

    Args:
        irreps_out (`e3nn_jax.Irreps`): output representations
        channel_out (optional int): if specified, the last axis before the irreps
            is assumed to be the channel axis and is mixed with the irreps.

    Example:
        >>> import e3nn_jax as e3nn
        >>> @hk.without_apply_rng
        ... @hk.transform
        ... def linear(x):
        ...     return e3nn.Linear("0e + 1o")(x)
        >>> x = e3nn.IrrepsArray("1o + 2x0e", jnp.ones(5))
        >>> params = linear.init(jax.random.PRNGKey(0), x)
        >>> y = linear.apply(params, x)
    """

    def __init__(
        self,
        irreps_out: Irreps,
        channel_out: int = None,
        *,
        irreps_in: Optional[Irreps] = None,
        biases: bool = False,
        path_normalization: Union[str, float] = None,
        gradient_normalization: Union[str, float] = None,
        get_parameter: Optional[Callable[[str, Instruction], jnp.ndarray]] = None,
    ):
        super().__init__()

        self.irreps_in = Irreps(irreps_in) if irreps_in is not None else None
        self.channel_out = channel_out
        self.irreps_out = Irreps(irreps_out)
        self.instructions = None
        self.biases = biases
        self.path_normalization = path_normalization
        self.gradient_normalization = gradient_normalization
        if get_parameter is None:

            def get_parameter(name: str, instruction: Instruction):
                return hk.get_parameter(
                    name,
                    shape=instruction.path_shape,
                    init=hk.initializers.RandomNormal(stddev=instruction.weight_std),
                )

        self.get_parameter = get_parameter

    def __call__(self, input: IrrepsArray) -> IrrepsArray:
        if self.irreps_in is not None:
            input = input._convert(self.irreps_in)

        input = input.remove_nones().simplify()
        output_irreps = self.irreps_out.simplify()
        if self.channel_out is not None:
            input = input.repeat_mul_by_last_axis()
            output_irreps = Irreps([(self.channel_out * mul, ir) for mul, ir in output_irreps])

        lin = FunctionalLinear(
            input.irreps,
            output_irreps,
            self.instructions,
            biases=self.biases,
            path_normalization=self.path_normalization,
            gradient_normalization=self.gradient_normalization,
        )
        w = [
            self.get_parameter(f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}", ins)
            if ins.i_in == -1
            else self.get_parameter(
                f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}", ins
            )
            for ins in lin.instructions
        ]

        f = lambda x: lin(w, x)  # kfac_jax.register_dense(lin(w, x), x, w)
        for _ in range(input.ndim - 1):
            f = jax.vmap(f)
        output = f(input)

        if self.channel_out is not None:
            output = output.factor_mul_to_last_axis(self.channel_out)
        return output._convert(self.irreps_out)


class NamedE3Linear(hk.Module):
    def __init__(self, target_irreps, keep_zero_outputs=False, with_bias=False, path_normalization=None, name=None):
        super().__init__(name=name)
        if isinstance(target_irreps, str):
            target_irreps = e3nn.Irreps(target_irreps)
        self.target_irreps = target_irreps
        self._keep_zero_outputs = keep_zero_outputs
        self.with_bias = with_bias
        self.path_normalization = path_normalization

    def __call__(self, x):
        if self._keep_zero_outputs:
            target_irreps = self.target_irreps
        else:
            target_irreps = self.target_irreps.filter(x.irreps)
        return Linear(
            target_irreps, path_normalization=self.path_normalization, gradient_normalization="element", biases=self.with_bias
        )(x)


def _normalize_function(phi):
    with jax.ensure_compile_time_eval():
        k = jax.random.PRNGKey(0)
        x = jax.random.normal(k, (1_000_000,))
        c = jnp.mean(phi(x) ** 2) ** 0.5

        if jnp.allclose(c, 1.0):
            return phi
        else:

            def rho(x):
                return phi(x) / c

            return rho


def norm_nonlinearity(x: e3nn.IrrepsArray, activation):
    """Computes the norm of all non-scalar inputs and uses them as gate values"""
    scalars = x.filtered(["0e", "0o"])
    vectors = x.filtered(lambda mul_ir: mul_ir.ir.l > 0)

    activation_scalar = _normalize_function(activation)
    activation_gate = _normalize_function(jax.nn.sigmoid)

    if scalars.shape[-1] != 0:
        scalar_irreps = scalars.irreps
        scalars = activation_scalar(scalars.array)
        scalars = e3nn.IrrepsArray(scalar_irreps, scalars)

    if vectors.shape[-1] != 0:
        vector_norms = e3nn.norm(vectors, squared=True)
        vector_norms = activation_gate(vector_norms.array)
        vector_norms = e3nn.IrrepsArray(f"{vector_norms.shape[-1]}x0e", vector_norms)
        vectors = vector_norms * vectors

    return e3nn.concatenate([scalars, vectors])


def _filter_scalar(mul_ir):
    return mul_ir.ir.l == 0


def _filter_vector(mul_ir):
    return mul_ir.ir.l > 0


def tile_e3(x: e3nn.IrrepsArray, n_tiles: Tuple[int]):
    return e3nn.IrrepsArray(x.irreps, jnp.tile(x.array, n_tiles))


def swapaxes_e3(x: e3nn.IrrepsArray, axis1: int, axis2: int):
    return e3nn.IrrepsArray(x.irreps, jnp.swapaxes(x.array, axis1, axis2))




class E3LinearChannelMixing(hk.Module):
    def __init__(self, n_channels_out = None, with_bias=True, name=None):
        super().__init__(name=name)
        self.n_channels_out = n_channels_out
        self.with_bias = with_bias

    def __call__(self, x: e3nn.IrrepsArray):
        n_channels_in = x.shape[-2]
        stddev = 1.0 / np.sqrt(n_channels_in)
        w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        n_channels_out = self.n_channels_out or x.shape[-2]

        y_out = []
        ind_start = 0
        for ir in x.irreps:
            for n in range(ir.mul):
                w = hk.get_parameter(f"w_{ir.ir}_{n}", [n_channels_out, n_channels_in], jnp.float32, w_init)
                x_slice = x.array[..., ind_start : ind_start + ir.ir.dim]
                ind_start += ir.ir.dim
                y = jnp.einsum("ij, ...jf->...if", w, x_slice)
                if ir.ir.l == 0 and self.with_bias:
                    b = hk.get_parameter(f"b_{ir.ir}_{n}", [n_channels_out], jnp.float32, jnp.zeros)
                    y += b[:, None]
                y_out.append(y)
                
        # concatenate across irreps
        return e3nn.IrrepsArray(x.irreps, jnp.concatenate(y_out, axis=-1))



class E3ChannelNorm(hk.Module):
    def __init__(self, trainable_scale=True, eps=1e-6, axis=-2, name=None):
        super().__init__(name=name)
        self.eps = eps
        self.axis = axis
        self.trainable_scale = trainable_scale


    def __call__(self, x: e3nn.IrrepsArray):
        x = x - e3nn.mean(x, axis=self.axis, keepdims=True)
        norm = e3nn.norm(x, squared=True).array
        scale = 1.0 / jnp.sqrt(jnp.mean(norm, axis=-2, keepdims=True) + self.eps)
        if self.trainable_scale:
            g = hk.get_parameter(f"scale", shape=[x.irreps.num_irreps], init=jnp.ones)
            scale = scale * g
        x = x * scale
        return x



def get_irreps_up_to_lmax(lmax, n_channels=1, as_string=False, all_parities=False):
    if all_parities:
        s = "+".join([f"{n_channels}x{l}e+{n_channels}x{l}o" for l in range(lmax + 1)])
    else:
        s = "+".join([f"{n_channels}x{l}{'o' if l%2 else 'e'}" for l in range(lmax + 1)])
    if as_string:
        return s
    else:
        return e3nn.Irreps(s)


def to_irreps_array(data):
    if isinstance(data, e3nn.IrrepsArray):
        return data
    else:
        return e3nn.IrrepsArray(f"{data.shape[-1]}x0e", data)


def e3stack(arrays: List[IrrepsArray], axis):
    irreps = arrays[0].irreps
    assert axis != -1, "Cannot stack along last dimension, since it is reserved for the irreps"
    for a in arrays:
        assert a.irreps == irreps
    output = jnp.stack([a.array for a in arrays], axis)
    return e3nn.IrrepsArray(irreps, output)


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    hidden_irreps = ["7x0e + 11x1o", "7x0e + 11x1o"]
    input_irreps = "5x0e + 4x1o"
    output_irreps = "1x1o"
    x = e3nn.normal(input_irreps, rng, leading_shape=(2,))

    model = hk.without_apply_rng(hk.transform(lambda x: E3LayerNorm()(x)))
    params = model.init(rng, x)
    y = model.apply(params, x)

    # x = x - e3nn.mean(x, axis=-1, keepdims=True)
    ind_start = 0

    # for row in x:
    #     print(row)

    # lin = FunctionalLinear(input_irreps, output_irreps)
    # model = hk.without_apply_rng(hk.transform(lambda x: E3MLP(hidden_irreps)(x)))
    # params = model.init(rng, x)
    # y = model.apply(params, x)
