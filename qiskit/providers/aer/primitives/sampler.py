# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Sampler class.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.primitives import SamplerResult
from qiskit.primitives.utils import init_circuit
from qiskit.result import QuasiDistribution

from .. import AerSimulator
from .base_sampler import BaseSampler  # TODO: fix import path after Terra 0.21.


class Sampler(BaseSampler):
    """
    Aer implementation of Sampler class.

    :Run Options:

        - **shots** (None or int) --
          The number of shots. If None, it calculates the probabilities.
          Otherwise, it samples from multinomial distributions.

        - **seed_primitive** (np.random.Generator or int) --
          Set a fixed seed or generator for rng. If shots is None, this option is ignored.
    """

    def __init__(
        self,
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        parameters: Iterable[Iterable[Parameter]] | None = None,
        backend_options: dict | None = None,
        transpile_options: dict | None = None,
    ):
        """
        Args:
            circuits: circuits to be executed.
            parameters: Parameters of each of the quantum circuits.
                Defaults to ``[circ.parameters for circ in circuits]``.
            backend_options: Options passed to AerSimulator.
            transpile_options: Options passed to transpile.
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        circuits = tuple(init_circuit(circuit) for circuit in circuits)

        super().__init__(
            circuits=circuits,
            parameters=parameters,
        )
        self._is_closed = False
        self._backend = AerSimulator()
        backend_options = {} if backend_options is None else backend_options
        self._backend.set_options(**backend_options)
        self._transpile_options = {} if transpile_options is None else transpile_options

    def _call(
        self,
        circuits: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

        seed_primitive = run_options.pop("seed_primitive", None)
        if seed_primitive is not None:
            seed_simulator = seed_primitive
            run_options.setdefault("seed_simulator", seed_simulator)

        experiments = []
        parameter_binds = []
        for i, value in zip(circuits, parameter_values):
            if len(value) != len(self._parameters[i]):
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(self._parameters[i])})."
                )

            circuit = self._circuits[i]
            experiments.append(circuit)
            parameter = {k: [v] for k, v in zip(self._parameters[i], value)}
            parameter_binds.append(parameter)

        experiments = transpile(experiments, self._backend, **self._transpile_options)
        result = self._backend.run(
            experiments, parameter_binds=parameter_binds, **run_options
        ).result()

        # Initialize metadata
        metadata = [{}] * len(circuits)
        quasis = []

        for i, meta in enumerate(metadata):
            counts = result.get_counts(i)
            shots = counts.shots()
            quasi = QuasiDistribution({k: v / shots for k, v in counts.items()})
            quasis.append(quasi)
            meta["shots"] = shots

        return SamplerResult(quasis, metadata)

    def close(self):
        self._is_closed = True
