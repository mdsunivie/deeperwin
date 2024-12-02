import copy

import jax.numpy as jnp
import jax
import numpy as np
import haiku as hk

from deeperwin.configuration import Configuration
from deeperwin.model.input_features import InputPreprocessor
from deeperwin.model.mlp import MLP
from deeperwin.model.embeddings.gnn_embedding import GNNEmbedding
from deeperwin.model.gnn import ScaleFeatures
from deeperwin.model.wavefunction import construct_wavefunction_definition
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "../configs"
CHECKPOINTS_DIR = Path(__file__).parent / "../checkpoints"


class ExpDecayFeatures(hk.Module):
    def __init__(
        self,
        mlp_config,
        input_config,
        exp_scaling_config,
        wavefunction_definition,
        name: str = "ExpDecayFeat",
    ) -> None:
        super().__init__(name=name)

        self.input_preprocessor = InputPreprocessor(
            config=input_config,
            mlp_config=mlp_config,
            wavefunction_definition=wavefunction_definition,
            name="input",
        )

        self.mlp_edge = MLP([10] * 1, linear_out=False)
        print(exp_scaling_config)
        self.scaler = ScaleFeatures(exp_scaling_config, 10, name="el_ion")

    def __call__(self, n_up: int, n_dn: int, r, R, Z):
        diff_dist, features = self.input_preprocessor(n_up, n_dn, r, R, Z, None)
        el_ion_edge = self.mlp_edge(features.el_ion)
        el_ion_edge *= self.scaler(diff_dist.dist_el_ion[..., None])

        return diff_dist.dist_el_ion, el_ion_edge


class Embedding(hk.Module):
    def __init__(
        self,
        mlp_config,
        input_config,
        embedding_config,
        wavefunction_definition,
        name="gnn_embedding",
    ):
        super().__init__(name=name)

        self.input_preprocessor = InputPreprocessor(
            config=input_config,
            mlp_config=mlp_config,
            wavefunction_definition=wavefunction_definition,
            name="input",
        )

        self.gnn = GNNEmbedding(embedding_config, mlp_config)

    def __call__(self, n_up, n_dn, r, R, Z):
        diff_dist, features = self.input_preprocessor(n_up, n_dn, r, R, Z, None)
        emb = self.gnn(diff_dist, features, n_up)

        return emb.el_el, emb.el_ion, emb.el


def test_edge_scaler():
    config_file = CONFIG_DIR / "config_exp_decay_edge.yml"
    raw_config, config = Configuration.load_configuration_file(config_file)
    assert isinstance(raw_config, dict)

    # geometry definition
    R, Z = jnp.array(config.physical.R), jnp.array(config.physical.Z)
    n_up, n_dn = config.physical.n_up, config.physical.n_dn

    ### build dummy model
    # init
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(1234), 2)
    wavefunction_definition = construct_wavefunction_definition(config.model, config.physical)

    # Build model
    def edge_features(n_up: int, n_dn: int, r, R, Z):
        return ExpDecayFeatures(
            config.model.mlp,
            config.model.features,
            config.model.embedding.exp_scaling,
            wavefunction_definition,
        )(n_up, n_dn, r, R, Z)

    r = jax.random.normal(rng1, [1, config.physical.n_electrons, 3])
    model = hk.transform(edge_features)
    params = model.init(rng2, n_up, n_dn, r, R, Z)

    edge_model = lambda params, r: model.apply(params, None, n_up, n_dn, r, R, Z)

    ### evaluate
    # test 1: init electron position close to nucleus
    # the edge features scaled by exp. based on the distance to the nucleus should not be zero,
    # representing a connection in the graph structure
    r1 = jax.random.normal(rng1, [1, config.physical.n_electrons, 3]) * 0.1  # close to the nucleus
    dist_el_ion, scaled_el_ion_feat = edge_model(params, r1)
    assert scaled_el_ion_feat.shape == (1, config.physical.n_electrons, 1, 10)
    assert not np.isclose(
        np.sum(np.abs(np.array(scaled_el_ion_feat[0, 0, :, :]))),
        0.0,
        atol=0.0001,  # batch x nb el x nuclei x feat dim
    )

    # test 2: move the electrons far away from the nucleus
    # far away from the nucleus there should be no edge
    r2 = r1 + 10  # close to the nucleus
    _, scaled_el_ion_feat_2 = edge_model(params, r2)
    assert np.isclose(
        np.sum(np.abs(np.array(scaled_el_ion_feat_2[0, 0, :, :]))), 0.0, atol=0.0001
    ), f"{np.sum(np.abs(np.array(scaled_el_ion_feat_2[0, 0, :, :])))}"


def test_gnn_edge_scaler():
    config_file = CONFIG_DIR / "config_exp_decay_edge.yml"
    raw_config, config = Configuration.load_configuration_file(config_file)
    assert isinstance(raw_config, dict)

    # geometry definition
    R, Z = jnp.array(config.physical.R), jnp.array(config.physical.Z)
    n_up, n_dn = config.physical.n_up, config.physical.n_dn

    ### build dummy model
    # init
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(1234), 2)
    wavefunction_definition = construct_wavefunction_definition(config.model, config.physical)

    # Build model
    def edge_embedding(n_up: int, n_dn: int, r, R, Z):
        return Embedding(
            config.model.mlp,
            config.model.features,
            config.model.embedding,
            wavefunction_definition,
        )(n_up, n_dn, r, R, Z)

    r = jax.random.normal(rng1, [1, config.physical.n_electrons, 3])
    model = hk.transform(edge_embedding)
    params = model.init(rng2, n_up, n_dn, r, R, Z)

    edge_embedding = lambda params, r: model.apply(params, None, n_up, n_dn, r, R, Z)

    ### evaluate
    r1 = jax.random.normal(rng1, [1, config.physical.n_electrons, 3]) * 0.1  # close to the nucleus
    edge_el_el, edge_el_ion, _ = edge_embedding(params, r1)

    assert not np.isclose(
        np.sum(np.abs(np.array(edge_el_el[0, 0, 2, :]))),
        0.0,
        atol=0.0001,  # batch x nb el x nuclei x feat dim
    )

    r2 = copy.deepcopy(r1)
    r2 = r2.at[0, 2, :].add(100)  # shift electron 3 far away
    dist = jnp.linalg.norm(r2[..., None, :, :] - r2[..., :, None, :], axis=-1)
    assert dist[0, 0, 2] > 95.0  # batch x n-el x n-el

    edge_el_el, edge_el_ion, emb_el_r2 = edge_embedding(params, r2)
    assert np.isclose(
        np.sum(np.abs(np.array(edge_el_ion[0, 2, 0, :]))),
        0.0,
        atol=0.0001,  # batch x nb el x nuclei x feat dim
    )
    # all electrons except the shifted electron [0, 1, 3, 4] - self interaction allowed
    assert np.isclose(
        np.sum(np.abs(np.array(edge_el_el[0, 2, [0, 1, 3, 4], :]))),
        0.0,
        atol=0.0001,  # batch x nb el x nuclei x feat dim
    ), np.sum(np.abs(np.array(edge_el_el[0, 2, [0, 1, 3, 4], :])))

    r3 = copy.deepcopy(r2)
    r3 = r3.at[0, 1, :].add(0.02)

    _, _, emb_el_r3 = edge_embedding(params, r3)

    assert np.linalg.norm(emb_el_r3[..., 2, :] - emb_el_r2[..., 2, :]) == 0
    assert np.linalg.norm(emb_el_r3[..., 1, :] - emb_el_r2[..., 1, :]) != 0
