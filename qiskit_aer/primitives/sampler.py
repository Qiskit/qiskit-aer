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
from qiskit.primitives.utils import final_measurement_mapping, init_circuit
from qiskit.result import QuasiDistribution

from .. import AerSimulator


class Sampler(BaseSampler):
    """
    Aer implementation of Sampler class.

    :Run Options:

        - **shots** (None or int) --
          The number of shots. If None, it calculates the probabilities exactly.
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

        self._cache = {}

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

        # Transpile circuits if not in cache
        no_cache_circuits = []
        for circuit_index in set(circuits):
            if circuit_index not in self._cache:
                no_cache_circuits.append(circuit_index)
        if no_cache_circuits:
            transpiled_circuits = (self._circuits[i] for i in no_cache_circuits)
            if "shots" in run_options and run_options["shots"] is None:
                transpiled_circuits = (
                    self._preprocess_circuit(circ) for circ in transpiled_circuits
                )
            if not self._skip_transpilation:
                transpiled_circuits = transpile(
                    list(transpiled_circuits),
                    self._backend,
                    **self._transpile_options,
                )
        for i, circuit in zip(no_cache_circuits, transpiled_circuits):
            self._cache[i] = circuit

        # Prepare circuits and parameter_binds
        experiments = []
        parameter_binds = []
        for i, value in zip(circuits, parameter_values):
            self._validate_parameter_length(value, i)
            parameter_binds.append({k: [v] for k, v in zip(self._parameters[i], value)})
            experiments.append(self._cache[i])

        result = self._backend.run(
            experiments, parameter_binds=parameter_binds, **run_options
        ).result()

        # Postprocessing
        metadata = []
        quasis = []
        for i in range(len(experiments)):
            if "shots" in run_options and run_options["shots"] is None:
                probabilities = result.data(i)["probabilities"]
                quasis.append(QuasiDistribution(probabilities))
                metadata.append({"shots": None, "simulator_metadata": result.results[i].metadata})
            else:
                counts = result.data(i)["counts"]
                shots = sum(counts.values())
                quasis.append(QuasiDistribution({k: v / shots for k, v in counts.items()}))
                metadata.append({"shots": shots, "simulator_metadata": result.results[i].metadata})

        return SamplerResult(quasis, metadata)

    def close(self):
        self._is_closed = True

    @staticmethod
    def _preprocess_circuit(circuit: QuantumCircuit):
        circuit = init_circuit(circuit)
        q_c_mapping = final_measurement_mapping(circuit)
        if set(range(circuit.num_clbits)) != set(q_c_mapping.values()):
            raise QiskitError(
                "Some classical bits are not used for measurements. "
                f"The number of classical bits {circuit.num_clbits}, "
                f"the used classical bits {set(q_c_mapping.values())}."
            )
        c_q_mapping = sorted((c, q) for q, c in q_c_mapping.items())
        qargs = [q for _, q in c_q_mapping]
        circuit = circuit.remove_final_measurements(inplace=False)
        circuit.save_probabilities_dict(qargs)
        return circuit

    def _validate_parameter_length(self, parameter, circuit_index):
        if len(parameter) != len(self._parameters[circuit_index]):
            raise QiskitError(
                f"The number of values ({len(parameter)}) does not match "
                f"the number of parameters ({len(self._parameters[circuit_index])})."
            )
