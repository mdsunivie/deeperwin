from typing import Tuple

import haiku as hk
from jax import numpy as jnp

from deeperwin.configuration import OrbitalsConfig, MLPConfig
from deeperwin.model import DiffAndDistances, Embeddings
from deeperwin.model.orbitals.transferable_atomic_orbitals import TransferableAtomicOrbitals
from deeperwin.model.orbitals.envelope_orbitals import EnvelopeOrbitals
from deeperwin.model.orbitals.periodic_envelope_orbitals import PeriodicEnvelopeOrbitals, BlochWaveEnvelope


class OrbitalNet(hk.Module):
    def __init__(
        self, config: OrbitalsConfig, mlp_config: MLPConfig, complex_wf: bool = False, name: str = None
    ) -> None:
        """
        NN representing any set of (spin) orbitals, e.g. EnvelopeOrbitals, BaselineOrbitals or a mixture of both

        * physical_config is passed down in order to calculate analytical initialization of the orbital
        exponential envelopes & determine how many up/dn orbitals should be outputted

        """
        super().__init__(name=name)
        self.config = config
        self.mlp_config = mlp_config
        self.full_det = (self.config.determinant_schema == "full_det") or (
            self.config.determinant_schema == "restricted_closed_shell"
        )
        self.complex_wf = complex_wf

        self.taos = None
        if self.config.transferable_atomic_orbitals:
            if self.config.transferable_atomic_orbitals.name == "taos":
                self.taos = TransferableAtomicOrbitals(
                    self.config.transferable_atomic_orbitals,
                    self.config.n_determinants,
                    self.config.determinant_schema,
                    complex_wf,
                )
            else:
                raise ValueError(
                    f"Unknown option for transferable_atomic_orbitals: {self.config.transferable_atomic_orbitals.name}"
                )

        self.periodic_env = None
        if self.config.periodic_orbitals:
            self.periodic_env = PeriodicEnvelopeOrbitals(
                self.config.periodic_orbitals,
                self.mlp_config,
                self.config.n_determinants,
                self.config.determinant_schema,
            )

        self.bloch_env = None
        if self.config.use_bloch_envelopes:
            self.bloch_env = BlochWaveEnvelope(self.config.determinant_schema)

    def __call__(
        self, diff_dist: DiffAndDistances, embeddings: Embeddings, fixed_params, n_ions: int, n_up: int, n_dn: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mo_up, mo_dn = 1.0, 1.0
        if self.config.envelope_orbitals:
            up, dn = EnvelopeOrbitals(
                self.config.envelope_orbitals,
                self.mlp_config,
                self.config.n_determinants,
                self.full_det,
                self.config.determinant_schema,
                self.complex_wf,
            )(diff_dist.dist_el_ion, embeddings.el, n_ions, n_up, n_dn)
            mo_up *= up
            mo_dn *= dn

        if self.taos:
            up, dn = self.taos(
                diff_dist,
                embeddings,
                n_ions,
                n_up,
                n_dn,
                fixed_params["transferable_atomic_orbitals"]["features"],
                fixed_params["cache"].get("taos") if fixed_params.get("cache") else None,
            )
            mo_up *= up
            mo_dn *= dn

        if self.bloch_env:
            up, dn = self.bloch_env(
                diff_dist.nonperiodic_diff_el_ion,
                fixed_params["baseline_orbitals"].k_points,
                fixed_params["periodic"].k_twist,
                n_up,
                n_dn,
            )
            mo_up *= up
            mo_dn *= dn

        if self.config.periodic_orbitals:
            k_points = fixed_params["periodic_orbitals"]["k_point_grid"]
            up, dn = self.periodic_env(diff_dist.nonperiodic_diff_el_ion, k_points, n_up, n_dn)
            mo_up *= up
            mo_dn *= dn

        if "periodic" in fixed_params:
            # Multiplying the twist on the orbitals instead of the full wavefunction is computationally less efficient,
            # but avoids having to keep track of multiplying it separately for the full wavefunction and the orbitals during pre-training
            el_pos = jnp.mean(diff_dist.nonperiodic_diff_el_ion, axis=-2)  # sum over ions
            phase = jnp.exp(1j * (el_pos @ fixed_params["periodic"].k_twist))
            mo_up *= phase[..., None, :n_up, None]  # [batch x det x el x orb]
            mo_dn *= phase[..., None, n_up:, None]
        return mo_up, mo_dn
