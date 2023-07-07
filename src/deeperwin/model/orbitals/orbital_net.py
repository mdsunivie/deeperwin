from typing import Tuple, Dict

import haiku as hk
from jax import numpy as jnp

from deeperwin.configuration import OrbitalsConfig, MLPConfig
from deeperwin.model import WavefunctionDefinition, DiffAndDistances, Embeddings
from deeperwin.model.orbitals.baseline_orbitals import BaselineOrbitals, _determine_baseline_output_shapes
from deeperwin.model.orbitals.transferable_atomic_orbitals import TransferableAtomicOrbitals
from deeperwin.model.orbitals.e3_transferable_atomic_orbitals import E3TransferableAtomicOrbitals
from deeperwin.model.orbitals.envelope_orbitals import EnvelopeOrbitals
from deeperwin.orbitals import OrbitalParams


class OrbitalNet(hk.Module):
    def __init__(
        self,
        config: OrbitalsConfig,
        mlp_config: MLPConfig,
        wavefunction_definition: WavefunctionDefinition,
        name: str = None
    ) -> None:
        """
        NN representing any set of (spin) orbitals, e.g. EnvelopeOrbitals, BaselineOrbitals or a mixture of both

        * physical_config is passed down in order to calculate analytical initialization of the orbital
        exponential envelopes & determine how many up/dn orbitals should be outputted

        """
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config
        self.wavefunction_definition = wavefunction_definition
        self.full_det = (self.config.determinant_schema == "full_det") or (self.config.determinant_schema == "restricted_closed_shell")

        self.taos = None
        self.e3_taos = None
        if self.config.transferable_atomic_orbitals and self.config.transferable_atomic_orbitals.name == "taos":
            self.taos = TransferableAtomicOrbitals(
                    self.config.transferable_atomic_orbitals,
                    self.mlp_config,
                    self.config.n_determinants,
                    self.config.determinant_schema,
                )
        elif self.config.transferable_atomic_orbitals and self.config.transferable_atomic_orbitals.name == "e3_taos":
            self.e3_taos = E3TransferableAtomicOrbitals(
                    self.config.transferable_atomic_orbitals,
                    self.config.n_determinants,
                    self.config.determinant_schema,
                )

    def _truncate_predicted_orbitals(
        self,
        mo_up: jnp.ndarray,
        mo_dn: jnp.ndarray
    ) -> Tuple[jnp.ndarray]:
        if self.full_det:
            n_el = mo_up.shape[-2] + mo_dn.shape[-2]
            mo_up = mo_up[..., :, :n_el]
            mo_dn = mo_dn[..., :, :n_el]
        else:
            mo_up = mo_up[..., :, :mo_up.shape[-2]]
            mo_dn = mo_dn[..., :, :mo_dn.shape[-2]]
        return mo_up, mo_dn

    def __call__(
        self,
        diff_dist: DiffAndDistances,
        embeddings: Embeddings,
        orbital_params: OrbitalParams,
        tao_params: Dict,
        cache: Dict,
        n_ions: int,
        n_up: int,
        n_dn: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.config.envelope_orbitals:
            mo_up, mo_dn = EnvelopeOrbitals(
                self.config.envelope_orbitals,
                self.mlp_config,
                self.config.n_determinants,
                self.wavefunction_definition,
                self.full_det,
                self.config.determinant_schema,
            )(diff_dist.dist_el_ion, embeddings.el, n_ions, n_up, n_dn)
        else:
            mo_up, mo_dn = 0.0, 0.0

        if self.config.baseline_orbitals:
            mo_up_baseline, mo_dn_baseline = BaselineOrbitals(
                self.config.baseline_orbitals,
                self.mlp_config,
                self.config.n_determinants,
                self.full_det,
                _determine_baseline_output_shapes(n_up, n_dn, self.config.determinant_schema)
            )(diff_dist, embeddings, orbital_params)
            mo_up += mo_up_baseline
            mo_dn += mo_dn_baseline

        if self.taos:
            mo_up_tao, mo_dn_tao = self.taos(diff_dist,
                                             embeddings,
                                             n_ions,
                                             n_up,
                                             n_dn,
                                             tao_params['features'],
                                             cache.get('taos') if cache else None)
            mo_up += mo_up_tao
            mo_dn += mo_dn_tao
        if self.e3_taos:
            mo_up_tao, mo_dn_tao = self.e3_taos(
                diff_dist,
                embeddings,
                n_ions,
                n_up,
                n_dn,
                tao_params['features_e3'],
                cache.get('e3_taos') if cache else None)
            mo_up += mo_up_tao
            mo_dn += mo_dn_tao

        if n_up < self.wavefunction_definition.max_n_up_orbitals or n_dn < self.wavefunction_definition.max_n_dn_orbitals:
            mo_up, mo_dn = self._truncate_predicted_orbitals(mo_up, mo_dn)

        return mo_up, mo_dn
