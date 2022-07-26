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
from qiskit.primitives import BaseSampler, SamplerResult
from qiskit.primitives.utils import init_circuit
from qiskit.result import QuasiDistribution

from .. import AerSimulator


class Sampler(BaseSampler):
    """
    Aer implementation of Sampler class.

    :Run Options:

        - **shots** (None or int) --
          The number of shots. If None, it calculates the probabilities.
          Otherwise, it samples from multinomial distributions.

        - **seed** (int) --
          Set a fixed seed for ``seed_simulator``. If shots is None, this option is ignored.

    .. note::
        Precedence of seeding is as follows:

        1. ``seed_simulator`` in runtime (i.e. in :meth:`__call__`)
        2. ``seed`` in runtime (i.e. in :meth:`__call__`)
        3. ``seed_simulator`` of ``backend_options``.
        4. default.
    """

    def __init__(
        self,
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        parameters: Iterable[Iterable[Parameter]] | None = None,
        backend_options: dict | None = None,
        transpile_options: dict | None = None,
        skip_transpilation: bool = False,
    ):
        """
        Args:
            circuits: Circuits to be executed.
            parameters: Parameters of each of the quantum circuits.
                Defaults to ``[circ.parameters for circ in circuits]``.
            backend_options: Options passed to AerSimulator.
            transpile_options: Options passed to transpile.
            skip_transpilation: if True, transpilation is skipped.
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
        self._skip_transpilation = skip_transpilation

    def _call(
        self,
        circuits: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

        seed = run_options.pop("seed", None)
        if seed is not None:
            run_options.setdefault("seed_simulator", seed)

        # Prepare circuits and parameter_binds
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

        # Transpile and Run
        if not self._skip_transpilation:
            experiments = transpile(experiments, self._backend, **self._transpile_options)
        result = self._backend.run(
            experiments, parameter_binds=parameter_binds, **run_options
        ).result()

        # Postprocessing
        metadata = []
        quasis = []
        for i in range(len(experiments)):
            counts = result.get_counts(i)
            shots = counts.shots()
            quasis.append(QuasiDistribution({k: v / shots for k, v in counts.items()}))
            metadata.append({"shots": shots, "simulator_metadata": result.results[i].metadata})

        return SamplerResult(quasis, metadata)

    def close(self):
        self._is_closed = True
