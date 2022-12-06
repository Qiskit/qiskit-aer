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
        circuits: QuantumCircuit | Iterable[QuantumCircuit] | None = None,
        parameters: Iterable[Iterable[Parameter]] | None = None,
        backend_options: dict | None = None,
        transpile_options: dict | None = None,
        run_options: dict | None = None,
        skip_transpilation: bool = False,
    ):
        """
        Args:
            circuits: Circuits to be executed.
            parameters: Parameters of each of the quantum circuits.
                Defaults to ``[circ.parameters for circ in circuits]``.
            backend_options: Options passed to AerSimulator.
            transpile_options: Options passed to transpile.
            run_options: Options passed to run.
            skip_transpilation: if True, transpilation is skipped.
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        if circuits is not None:
            circuits = tuple(init_circuit(circuit) for circuit in circuits)

        super().__init__(
            circuits=circuits,
            parameters=parameters,
            options=run_options,
        )
        self._is_closed = False
        self._backend = AerSimulator()
        backend_options = {} if backend_options is None else backend_options
        self._backend.set_options(**backend_options)
        self._transpile_options = {} if transpile_options is None else transpile_options
        self._skip_transpilation = skip_transpilation

        self._transpiled_circuits = {}

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

        is_shots_none = "shots" in run_options and run_options["shots"] is None
        self._transpile(circuits, is_shots_none)

        experiments = []
        parameter_binds = []
        for i, value in zip(circuits, parameter_values):
            if len(value) != len(self._parameters[i]):
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(self._parameters[i])})."
                )
            parameter_binds.append({k: [v] for k, v in zip(self._parameters[i], value)})
            experiments.append(self._transpiled_circuits[(i, is_shots_none)])

        result = self._backend.run(
            experiments, parameter_binds=parameter_binds, **run_options
        ).result()

        # Postprocessing
        metadata = []
        quasis = []
        for i in range(len(experiments)):
            if is_shots_none:
                probabilities = result.data(i)["probabilities"]
                num_qubits = result.results[i].metadata["num_qubits"]
                quasi_dist = QuasiDistribution(
                    {f"{k:0{num_qubits}b}": v for k, v in probabilities.items()}
                )
                quasis.append(quasi_dist)
                metadata.append({"shots": None, "simulator_metadata": result.results[i].metadata})
            else:
                counts = result.get_counts(i)
                shots = sum(counts.values())
                quasis.append(
                    QuasiDistribution(
                        {k: v / shots for k, v in counts.items()},
                        shots=shots,
                    )
                )
                metadata.append({"shots": shots, "simulator_metadata": result.results[i].metadata})

        return SamplerResult(quasis, metadata)

    # This method will be used after Terra 0.22.
    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ):
        # pylint: disable=no-name-in-module, import-error, import-outside-toplevel, no-member
        from typing import List

        from qiskit.primitives.primitive_job import PrimitiveJob
        from qiskit.primitives.utils import _circuit_key

        circuit_indices: List[int] = []
        for circuit in circuits:
            index = self._circuit_ids.get(_circuit_key(circuit))
            if index is not None:
                circuit_indices.append(index)
            else:
                circuit_indices.append(len(self._circuits))
                self._circuit_ids[_circuit_key(circuit)] = len(self._circuits)
                self._circuits.append(circuit)
                self._parameters.append(circuit.parameters)
        job = PrimitiveJob(self._call, circuit_indices, parameter_values, **run_options)
        job.submit()
        return job

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

    def _transpile(self, circuit_indices: Sequence[int], is_shots_none: bool):
        to_handle = [
            i for i in set(circuit_indices) if (i, is_shots_none) not in self._transpiled_circuits
        ]
        if to_handle:
            circuits = (self._circuits[i] for i in to_handle)
            if is_shots_none:
                circuits = (self._preprocess_circuit(circ) for circ in circuits)
            if not self._skip_transpilation:
                circuits = transpile(
                    list(circuits),
                    self._backend,
                    **self._transpile_options,
                )
            for i, circuit in zip(to_handle, circuits):
                self._transpiled_circuits[(i, is_shots_none)] = circuit
