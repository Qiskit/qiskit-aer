# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Estimator V2 implementation for Aer."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives.containers import DataBin, EstimatorPubLike, PrimitiveResult, PubResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.quantum_info import Pauli

from qiskit_aer import AerSimulator


@dataclass
class Options:
    """Options for :class:`~.EstimatorV2`."""

    default_precision: float = 0.0
    """The default precision to use if none are specified in :meth:`~run`.
    """

    backend_options: dict = field(default_factory=dict)
    """backend_options: Options passed to AerSimulator."""

    run_options: dict = field(default_factory=dict)
    """run_options: Options passed to run."""


class EstimatorV2(BaseEstimatorV2):
    """Evaluates expectation values for provided quantum circuit and observable combinations.

    Run a fast simulation using Aer.
    Sampling from a normal distribution ``N(expval, precison)`` when set to ``precision``.

    * ``default_precision``: The default precision to use if none are specified in :meth:`~run`.
      Default: 0.0.

    * ``backend_options``: Options passed to AerSimulator.
      Default: {}.

    * ``run_options``: Options passed to :meth:`AerSimulator.run`.
      Default: {}.
    """

    def __init__(
        self,
        *,
        options: dict | None = None,
    ):
        """
        Args:
            options: The options to control the default precision (``default_precision``),
                the backend options (``backend_options``), and
                the runtime options (``run_options``).
        """
        self._options = Options(**options) if options else Options()
        self._backend = AerSimulator(**self.options.backend_options)

    @classmethod
    def from_backend(cls, backend, **options):
        """make new sampler that uses external backend"""
        estimator = cls(**options)
        if isinstance(backend, AerSimulator):
            estimator._backend = backend
        else:
            estimator._backend = AerSimulator.from_backend(backend)
        return estimator

    @property
    def options(self) -> Options:
        """Return the options"""
        return self._options

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if precision is None:
            precision = self._options.default_precision
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        self._validate_pubs(coerced_pubs)
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _validate_pubs(self, pubs: list[EstimatorPub]):
        for i, pub in enumerate(pubs):
            if pub.precision < 0.0:
                raise ValueError(
                    f"The {i}-th pub has precision less than 0 ({pub.precision}). ",
                    "But precision should be equal to or larger than 0.",
                )

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        return PrimitiveResult([self._run_pub(pub) for pub in pubs], metadata={"version": 2})

    def _run_pub(self, pub: EstimatorPub) -> PubResult:
        circuit = pub.circuit.copy()
        observables = pub.observables
        parameter_values = pub.parameter_values
        precision = pub.precision

        # calculate broadcasting of parameters and observables
        param_shape = parameter_values.shape
        param_indices = np.fromiter(np.ndindex(param_shape), dtype=object).reshape(param_shape)
        bc_param_ind, bc_obs = np.broadcast_arrays(param_indices, observables)

        parameter_binds = {}
        param_array = parameter_values.as_array(circuit.parameters)
        parameter_binds = {p: param_array[..., i].ravel() for i, p in enumerate(circuit.parameters)}

        # save expval
        paulis = {pauli for obs_dict in observables.ravel() for pauli in obs_dict.keys()}
        for pauli in paulis:
            circuit.save_expectation_value(
                Pauli(pauli), qubits=range(circuit.num_qubits), label=pauli
            )
        result = self._backend.run(
            circuit, parameter_binds=[parameter_binds], **self.options.run_options
        ).result()

        # calculate expectation values (evs) and standard errors (stds)
        flat_indices = list(param_indices.ravel())
        evs = np.zeros_like(bc_param_ind, dtype=float)
        stds = np.full(bc_param_ind.shape, precision)
        for index in np.ndindex(*bc_param_ind.shape):
            param_index = bc_param_ind[index]
            flat_index = flat_indices.index(param_index)
            for pauli, coeff in bc_obs[index].items():
                expval = result.data(flat_index)[pauli]
                evs[index] += expval * coeff
        if precision > 0:
            rng = np.random.default_rng(self.options.run_options.get("seed_simulator"))
            if not np.all(np.isreal(evs)):
                raise ValueError("Given operator is not Hermitian and noise cannot be added.")
            evs = rng.normal(evs, precision, evs.shape)
        return PubResult(
            DataBin(evs=evs, stds=stds, shape=evs.shape),
            metadata={
                "target_precision": precision,
                "circuit_metadata": pub.circuit.metadata,
                "simulator_metadata": result.metadata,
            },
        )
