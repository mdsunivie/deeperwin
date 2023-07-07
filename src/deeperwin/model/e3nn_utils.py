from math import sqrt
from typing import Callable, List, NamedTuple, Optional, Tuple, Union
import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irreps, IrrepsArray, config
from e3nn_jax._src.core_tensor_product import _sum_tensors
from jax import numpy as jnp
from deeperwin.model.mlp import get_activation, get_rbf_features, MLP
from deeperwin.configuration import E3MACEGNNConfig, MLPConfig


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
            _sum_tensors(
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


def _tile_channels(irreps: e3nn.Irreps, n: int, as_string=False):
    irreps = [e3nn.MulIrrep(ir.mul * n, ir.ir) for ir in e3nn.Irreps(irreps)]
    irreps = e3nn.Irreps(irreps)
    if as_string:
        return str(irreps)
    else:
        return irreps


class SymmetricTensorProductLayer(hk.Module):
    def __init__(self, n_channels, l_max_linear, L_max_prod, max_order=2, has_channel_dim=False, name=None):
        super().__init__(name=name)
        irreps_lmax = get_irreps_up_to_lmax(l_max_linear, n_channels)
        irreps_Lmax = get_irreps_up_to_lmax(L_max_prod, 1)
        orders = [i + 1 for i in range(max_order)]
        self.has_channel_dim = has_channel_dim
        if has_channel_dim:
            self.lin_a = E3LinearChannelMixing(n_channels, name="channel_mixing")
        else:
            self.lin_a = NamedE3Linear(irreps_lmax, name="lin_a")
        self.tp = e3nn.haiku.SymmetricTensorProduct(orders, keep_irrep_out=irreps_Lmax)

    def __call__(self, x):
        a = self.lin_a(x)
        if not self.has_channel_dim:
            a = a.mul_to_axis()
        y = self.tp(a)
        if not self.has_channel_dim:
            y = y.axis_to_mul()
        return y


class GatedE3Activation(hk.Module):
    def __init__(
        self,
        scalar_activation: Union[Callable, str] = "silu",
        gate_activation_even: Union[Callable, str] = "silu",
        gate_activation_odd: Union[Callable, str] = "tanh",
        name=None,
    ):
        super().__init__(name=name)
        self._scalar_activation = get_activation(scalar_activation)
        self._gate_activation_even = get_activation(gate_activation_even)
        self._gate_activation_odd = get_activation(gate_activation_odd)

    def __call__(self, x):
        scalars = x.filtered(_filter_scalar)
        for irreps in scalars.irreps:
            assert irreps.ir.p == 1, "Gate activation only implemented for even (i.e. 0e) scalars"

        scalars_out = e3nn.IrrepsArray(scalars.irreps, self._gate_activation_even(scalars.array))
        vectors = x.filtered(_filter_vector)
        if vectors.shape[-1] > 0:
            gates = NamedE3Linear(f"{vectors.irreps.num_irreps}x0e", "gates")(scalars)
            gates = e3nn.IrrepsArray(gates.irreps, self._scalar_activation(gates.array))
            vectors_out = gates * vectors
            y = e3nn.concatenate([scalars_out, vectors_out], axis=-1)
        else:
            y = scalars_out
        return y


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


class E3MACEGNN(hk.Module):
    def __init__(
        self, config: E3MACEGNNConfig, input_has_channels=False, input_irreps=None, name: Optional[str] = None
    ):
        super().__init__(name)
        self.config = config
        self.input_has_channels = input_has_channels
        self.irreps_lmax_no_channel = get_irreps_up_to_lmax(self.config.l_max, 1)
        self.irreps_lmax_with_channel = get_irreps_up_to_lmax(self.config.l_max, self.config.n_channels)

    def __call__(self, node_features, edge_diff, edge_dist):
        Y = e3nn.spherical_harmonics(self.irreps_lmax_no_channel, edge_diff, normalize=False)
        rbfs = self._get_radial_embedding(edge_dist)

        for n in range(self.config.n_iterations):
            # Linear mixing of channels
            if n == 0 and not self.input_has_channels:
                node_features = NamedE3Linear(self.irreps_lmax_with_channel, name=f"lin_{n}")(node_features)
                node_features = node_features.mul_to_axis()
            else:
                node_features = E3LinearChannelMixing(self.config.n_channels, name=f"channel_mixing_{n}")(node_features)

            # Tensor product of node features with spherical harmonics of edge direction to obtain edge features
            edge: e3nn.IrrepsArray = e3nn.tensor_product(
                Y[..., None, :],  # add dummy dim for channel
                node_features[..., None, :, :, :],  # add dummy dim for receiving node
                filter_ir_out=self.irreps_lmax_no_channel,
            )
            n_irreps = edge.irreps.num_irreps

            # Weighting of edge features by radial basis functions
            radial_weights = hk.Linear(self.config.n_channels * n_irreps, name=f"radial_weights_{n}")(rbfs)
            radial_weights = radial_weights.reshape(radial_weights.shape[:-1] + (self.config.n_channels, -1))
            radial_weights = e3nn.IrrepsArray(f"{n_irreps}x0e", radial_weights)
            edge = radial_weights * edge
            edge = e3nn.sum(edge, axis=-1)

            # Sum over all sending nodes to obtain node features and apply symmetric tensor product
            node_features = e3nn.sum(
                edge, axis=-3
            )  # sum over all sending nodes (-1 = lm, -2 = channels, -4 = receiving node)
            node_features = SymmetricTensorProductLayer(
                self.config.n_channels, self.config.l_max, self.config.L_max, has_channel_dim=True, name=f"tp_{n}"
            )(node_features)

        if not self.input_has_channels:
            node_features = node_features.axis_to_mul()
        return node_features

    def _get_radial_embedding(self, edge_dist):
        rbfs = get_rbf_features(edge_dist, self.config.n_rbf, self.config.rbf_max_dist)
        rbfs = MLP([self.config.radial_mlp_width] * self.config.radial_mlp_n_layers, MLPConfig(), name="radial_mlp")(
            rbfs
        )
        return rbfs


class GVPLayer(hk.Module):
    def __init__(
        self,
        irreps_out,
        scalar_activation="silu",
        gate_activation_even="silu",
        gate_activation_odd="tanh",
        activate=True,
        name=None,
    ):
        super().__init__(name=name)
        irreps_out = e3nn.Irreps(irreps_out)
        self._irreps_out_scalar = irreps_out.filter(_filter_scalar)
        self._irreps_out_vector = irreps_out.filter(_filter_vector)
        self._scalar_activation = get_activation(scalar_activation)
        self._gate_activation_even = get_activation(gate_activation_even)
        self._gate_activation_odd = get_activation(gate_activation_odd)
        self._activate = activate

    def __call__(self, x: e3nn.IrrepsArray):
        scalars = x.filtered(_filter_scalar)
        vectors = x.filtered(_filter_vector)

        lin_for_norm = NamedE3Linear(vectors.irreps, "lin_vector_norm")(vectors)
        norms = e3nn.norm(lin_for_norm, squared=True)
        merged_scalars = e3nn.concatenate([scalars, norms], axis=-1)
        scalars_out = NamedE3Linear(self._irreps_out_scalar, "lin_scalar_out")(merged_scalars)
        if self._activate:
            scalars_out = e3nn.IrrepsArray(scalars_out.irreps, self._scalar_activation(scalars_out.array))

        if self._irreps_out_vector.dim > 0:
            vectors_out = NamedE3Linear(self._irreps_out_vector, "lin_vector_out")(vectors)
            if self._activate:
                gates = NamedE3Linear(f"{self._irreps_out_vector.num_irreps}x0e", "lin_vector_gates")(merged_scalars)
                vectors_out = e3nn.gate(
                    e3nn.concatenate([gates, vectors_out]),
                    self._scalar_activation,
                    None,
                    self._gate_activation_even,
                    self._gate_activation_odd,
                )
            x = e3nn.concatenate([scalars_out, vectors_out], axis=-1)
        else:
            x = scalars_out
        return x


class GVP(hk.Module):
    def __init__(
        self,
        irreps: List[Union[Irreps, str]],
        activate_final=True,
        activation_even="silu",
        activation_odd="tanh",
        name=None,
    ):
        super().__init__(name=name)
        self.irreps = [e3nn.Irreps(ir) for ir in irreps]
        self.activate_final = activate_final
        self.activation_even = get_activation(activation_even)
        self.activation_odd = get_activation(activation_odd)

    def __call__(self, x: e3nn.IrrepsArray):
        n_layers = len(self.irreps)
        for l, ir in enumerate(self.irreps):
            activate = (l != (n_layers - 1)) or self.activate_final
            x = GVPLayer(
                ir,
                scalar_activation=self.activation_even,
                gate_activation_even=self.activation_even,
                gate_activation_odd=self.activation_odd,
                activate=activate,
            )(x)
        return x


class TensorProductNet(hk.Module):
    def __init__(
        self,
        n_layers: int,
        l_max: int,
        n_channels: int,
        order: int = 2,
        l_max_out=None,
        use_activation=True,
        activate_final=True,
        use_layer_norm=False,
        scalar_activation="silu",
        gate_activation_even: str = "silu",
        gate_activation_odd: str = "tanh",
        name=None,
    ):
        super().__init__(name=name)
        self.use_layer_norm = use_layer_norm
        self.tp_layers = []
        self.activations = []
        if l_max_out is None:
            l_max_out = l_max
        for i in range(n_layers):
            not_last_layer = i != (n_layers - 1)
            self.tp_layers.append(
                SymmetricTensorProductLayer(
                    n_channels, l_max, l_max if not_last_layer else l_max_out, order, name=f"tp_{i}"
                )
            )
            if use_activation and (activate_final or not_last_layer):
                self.activations.append(
                    GatedE3Activation(
                        scalar_activation, gate_activation_even, gate_activation_odd, name=f"activation_{i}"
                    )
                )
            else:
                self.activations.append(None)

    def __call__(self, x: e3nn.IrrepsArray):
        for layer, (tp, activation) in enumerate(zip(self.tp_layers, self.activations)):
            x = tp(x)
            if activation:
                x = activation(x)
            if self.use_layer_norm:
                x = E3LayerNorm(name=f"LN_{layer}")(x)
        return x


class E3LayerNorm(hk.Module):
    def __init__(self, trainable_scale=True, eps=1e-5, name=None):
        super().__init__(name=name)
        self.trainable_scale = trainable_scale
        self.eps = 1e-5

    def __call__(self, x: e3nn.IrrepsArray):
        output_data = []
        ind_start = 0
        for ir in x.irreps:
            ind_end = ind_start + ir.dim
            data = x.array[..., ind_start:ind_end]
            ind_start = ind_start
            if ir.mul > 1:
                data = data.reshape(data.shape[:-1] + (ir.mul, -1))
                data -= data.mean(axis=-2, keepdims=True)
                scale = jnp.sqrt(jnp.sum(data**2, axis=[-1, -2], keepdims=True)) + self.eps
                if self.trainable_scale:
                    g = hk.get_parameter(f"scale_{ir.ir}", shape=[ir.mul], init=jnp.ones)
                    data = data * (g[:, None] / scale)
                else:
                    data /= scale
                data = data.reshape(data.shape[:-2] + (-1,))
                output_data.append(data)
            else:
                output_data.append(data)
        return e3nn.IrrepsArray(x.irreps, jnp.concatenate(output_data, axis=-1))


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



# class E3MLP(hk.Module):
#     def __init__(self, irreps: List[Union[Irreps, str]], activate_final=True, activation = "silu", name=None):
#         super().__init__(name=name)
#         self.irreps = [e3nn.Irreps(ir) for ir in irreps]
#         self.activate_final = activate_final
#         self.activation = get_activation(activation)
#
#     def __call__(self, x: e3nn.IrrepsArray):
#         n_layers = len(self.irreps)
#         for l, ir in enumerate(self.irreps):
#             x = NamedE3Linear(ir, f"e3_linear_{l}")(x)
#             if (l != (n_layers - 1)) or self.activate_final:
#                 x = norm_nonlinearity(x, self.activation)
#         return x


def symmetrize_e3_func(f, tmp_axis=-2):
    """
    Take an arbitrary function f and return a new function g(x) = f(x) + f(-x).

    For any continous f, the new function g is now symmetric, i.e. g(x) = g(-x)
    """

    def symm_f(x):
        x = e3nn.stack([x, x * (-1)], axis=tmp_axis)
        y = f(x)
        return e3nn.sum(y, axis=tmp_axis)

    return symm_f


def antisymmetrize_e3_func(f):
    """
    Take an arbitrary function f and return a new function g(x) = f(x) - f(-x).

    For any continous f, the new function g is now anti-symmetric, i.e. g(x) = -g(-x)
    """

    def asymm_f(x):
        x = e3nn.stack([x, x * (-1)], axis=-2)
        y = f(x)
        return y[..., 0, :] - y[..., 1, :]

    return asymm_f


def _get_ao_to_irreps_matrix_l2():
    s15 = np.sqrt(15)
    s5 = np.sqrt(5)
    s5_2 = s5 * 0.5
    s15_2 = s15 * 0.5
    U_matrix_2 = np.array(
        [
            [1, 0, 0, -s5_2, 0, -s15_2],
            [0, 0, s15, 0, 0, 0],
            [0, s15, 0, 0, 0, 0],
            [1, 0, 0, s5, 0, 0],
            [0, 0, 0, 0, s15, 0],
            [1, 0, 0, -s5_2, 0, s15_2],
        ]
    )
    return U_matrix_2


def get_ao_to_irreps_matrix(atomic_orbitals):
    """
    Generates a matrix U that maps from cartesian basis functions to irreducible representations.

    mo_coeffs, atomic_orbitals = result_of_hf_calculation() # shape [n_basis_funcs x orbitals]
    U, irreps = get_ao_to_irrpes_matrix(atomic_orbitals)
    mo_ir = e3nn.IrrepsArray(irreps, mo_coeffs.T @ ao_to_irreps_mat)

    Args:
        atomic_orbitals:

    Returns:
        U: Matrix of shape [n_basis_funcs x n_basis_funcs]
        irreps: str
    """
    irreps_out = []
    n_orbitals = len(atomic_orbitals)
    U = np.eye(n_orbitals)
    ind_ao = 0
    while ind_ao < len(atomic_orbitals):
        l = np.sum(atomic_orbitals[ind_ao].angular_momenta)
        if l == 0:
            irreps_out.append("1x0e")
            ind_ao += 1
        elif l == 1:
            irreps_out.append("1x1o")
            ind_ao += 3
        elif l == 2:
            U[ind_ao : ind_ao + 6, ind_ao : ind_ao + 6] = _get_ao_to_irreps_matrix_l2()
            irreps_out.append("1x0e+1x2e")
            ind_ao += 6
    irreps_out = "+".join(irreps_out)
    return U, irreps_out


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

def get_p_orbital_singular_values(x: e3nn.IrrepsArray, compute_uv=False):
    x = x.filter("1x1o")
    x = x.array.reshape(x.shape[:-1] + (-1, 3))
    return jnp.linalg.svd(x, compute_uv=compute_uv)


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
