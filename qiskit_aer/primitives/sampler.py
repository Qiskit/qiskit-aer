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

from collections.abc import Sequence, Iterable

from warnings import warn
import numpy as np
from qiskit.circuit import ParameterExpression, QuantumCircuit, Qubit
from qiskit.circuit.library.data_preparation import Initialize
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.primitives import BaseSamplerV1, SamplerResult
from qiskit.quantum_info import Statevector
from qiskit.result import QuasiDistribution

from .. import AerSimulator


def init_circuit(state: QuantumCircuit | Statevector) -> QuantumCircuit:
    """Initialize state by converting the input to a quantum circuit.

    Args:
        state: The state as quantum circuit or statevector.

    Returns:
        The state as quantum circuit.
    """
    if isinstance(state, QuantumCircuit):
        return state
    if not isinstance(state, Statevector):
        state = Statevector(state)
    qc = QuantumCircuit(state.num_qubits)
    qc.append(Initialize(state), qargs=range(state.num_qubits))
    return qc


def final_measurement_mapping(circuit: QuantumCircuit) -> dict[int, int]:
    """Return the final measurement mapping for the circuit.

    Dict keys label measured qubits, whereas the values indicate the
    classical bit onto which that qubits measurement result is stored.

    Parameters:
        circuit: Input quantum circuit.

    Returns:
        Mapping of qubits to classical bits for final measurements.
    """
    active_qubits = list(range(circuit.num_qubits))
    active_cbits = list(range(circuit.num_clbits))

    # Find final measurements starting in back
    mapping = {}
    for item in circuit._data[::-1]:
        if item.operation.name == "measure":
            cbit = circuit.find_bit(item.clbits[0]).index
            qbit = circuit.find_bit(item.qubits[0]).index
            if cbit in active_cbits and qbit in active_qubits:
                mapping[qbit] = cbit
                active_cbits.remove(cbit)
                active_qubits.remove(qbit)
        elif item.operation.name not in ["barrier", "delay"]:
            for qq in item.qubits:
                _temp_qubit = circuit.find_bit(qq).index
                if _temp_qubit in active_qubits:
                    active_qubits.remove(_temp_qubit)

        if not active_cbits or not active_qubits:
            break

    # Sort so that classical bits are in numeric order low->high.
    mapping = dict(sorted(mapping.items(), key=lambda item: item[1]))
    return mapping


def _bits_key(bits: tuple[Qubit, ...], circuit: QuantumCircuit) -> tuple:
    return tuple(
        (
            circuit.find_bit(bit).index,
            tuple((reg[0].size, reg[0].name, reg[1]) for reg in circuit.find_bit(bit).registers),
        )
        for bit in bits
    )


def _format_params(param):
    if isinstance(param, np.ndarray):
        return param.data.tobytes()
    elif isinstance(param, QuantumCircuit):
        return _circuit_key(param)
    elif isinstance(param, Iterable):
        return tuple(param)
    return param


def _circuit_key(circuit: QuantumCircuit, functional: bool = True) -> tuple:
    """Private key function for QuantumCircuit.

    This is the workaround until :meth:`QuantumCircuit.__hash__` will be introduced.
    If key collision is found, please add elements to avoid it.

    Args:
        circuit: Input quantum circuit.
        functional: If True, the returned key only includes functional data (i.e. execution related).

    Returns:
        Composite key for circuit.
    """
    functional_key: tuple = (
        circuit.num_qubits,
        circuit.num_clbits,
        circuit.num_parameters,
        tuple(  # circuit.data
            (
                _bits_key(data.qubits, circuit),  # qubits
                _bits_key(data.clbits, circuit),  # clbits
                data.operation.name,  # operation.name
                tuple(_format_params(param) for param in data.operation.params),  # operation.params
            )
            for data in circuit.data
        ),
        None if circuit._op_start_times is None else tuple(circuit._op_start_times),
    )
    if functional:
        return functional_key
    return (
        circuit.name,
        *functional_key,
    )


class Sampler(BaseSamplerV1):
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
        *,
        backend_options: dict | None = None,
        transpile_options: dict | None = None,
        run_options: dict | None = None,
        skip_transpilation: bool = False,
    ):
        """
        Args:
            backend_options: Options passed to AerSimulator.
            transpile_options: Options passed to transpile.
            run_options: Options passed to run.
            skip_transpilation: if True, transpilation is skipped.
        """
        warn(
            "Sampler has been deprecated as of Aer 0.15, please use SamplerV2 instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        super().__init__(options=run_options)
        # These two private attributes used to be created by super, but were deprecated in Qiskit
        # 0.46. See https://github.com/Qiskit/qiskit/pull/11051
        self._circuits = []
        self._parameters = []

        self._backend = AerSimulator()
        backend_options = {} if backend_options is None else backend_options
        self._backend.set_options(**backend_options)
        self._transpile_options = {} if transpile_options is None else transpile_options
        self._skip_transpilation = skip_transpilation

        self._transpiled_circuits = {}
        self._circuit_ids = {}

    def _call(
        self,
        circuits: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        seed = run_options.pop("seed", None)
        if seed is not None:
            run_options.setdefault("seed_simulator", seed)

        is_shots_none = "shots" in run_options and run_options["shots"] is None
        self._transpile(circuits, is_shots_none)

        experiment_manager = _ExperimentManager()
        for i, value in zip(circuits, parameter_values):
            if len(value) != len(self._parameters[i]):
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(self._parameters[i])})."
                )

            experiment_manager.append(
                key=i,
                parameter_bind=dict(zip(self._parameters[i], value)),
                experiment_circuit=self._transpiled_circuits[(i, is_shots_none)],
            )

        result = self._backend.run(
            experiment_manager.experiment_circuits,
            parameter_binds=experiment_manager.parameter_binds,
            **run_options,
        ).result()

        # Postprocessing
        metadata = []
        quasis = []
        for i in experiment_manager.experiment_indices:
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
                        {k.replace(" ", ""): v / shots for k, v in counts.items()},
                        shots=shots,
                    )
                )
                metadata.append({"shots": shots, "simulator_metadata": result.results[i].metadata})

        return SamplerResult(quasis, metadata)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ):
        # pylint: disable=no-name-in-module, import-error, import-outside-toplevel, no-member
        from typing import List

        from qiskit.primitives.primitive_job import PrimitiveJob

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
        # The public submit method was removed in Qiskit 0.46
        (job.submit if hasattr(job, "submit") else job._submit)()  # pylint: disable=no-member
        return job

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

    def _transpile_circuit(self, circuit):
        self._backend.set_max_qubits(circuit.num_qubits)
        transpiled = transpile(
            circuit,
            self._backend,
            **self._transpile_options,
        )
        return transpiled

    def _transpile(self, circuit_indices: Sequence[int], is_shots_none: bool):
        to_handle = [
            i for i in set(circuit_indices) if (i, is_shots_none) not in self._transpiled_circuits
        ]
        if to_handle:
            circuits = (self._circuits[i] for i in to_handle)
            if is_shots_none:
                circuits = (self._preprocess_circuit(circ) for circ in circuits)
            if not self._skip_transpilation:
                circuits = (self._transpile_circuit(circ) for circ in circuits)
            for i, circuit in zip(to_handle, circuits):
                self._transpiled_circuits[(i, is_shots_none)] = circuit


class _ExperimentManager:
    def __init__(self):
        self.keys: list[int] = []
        self.experiment_circuits: list[QuantumCircuit] = []
        self.parameter_binds: list[dict[ParameterExpression, list[float]]] = []
        self._input_indices: list[list[int]] = []
        self._num_experiment: int = 0

    def __len__(self):
        return self._num_experiment

    @property
    def experiment_indices(self):
        """indices of experiments"""
        return np.argsort(sum(self._input_indices, [])).tolist()

    def append(
        self,
        key: tuple[int, int],
        parameter_bind: dict[ParameterExpression, float],
        experiment_circuit: QuantumCircuit,
    ):
        """append experiments"""
        if parameter_bind and key in self.keys:
            key_index = self.keys.index(key)
            for k, vs in self.parameter_binds[key_index].items():
                vs.append(parameter_bind[k])
            self._input_indices[key_index].append(self._num_experiment)
        else:
            self.experiment_circuits.append(experiment_circuit)
            self.keys.append(key)
            self.parameter_binds.append({k: [v] for k, v in parameter_bind.items()})
            self._input_indices.append([self._num_experiment])

        self._num_experiment += 1
