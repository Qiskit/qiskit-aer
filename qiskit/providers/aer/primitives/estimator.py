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
Estimator class.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import accumulate

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.primitives.utils import init_circuit, init_observable
from qiskit.quantum_info import Pauli
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit.providers import Options

from .. import AerSimulator


class Estimator(BaseEstimator):
    """
    Aer implmentation of Estimator.

    :Run Options:
        - **shots** (None or int) --
          The number of shots. If None and approximation is True, it calculates the exact
          expectation values. Otherwise, it calculates expectation values with sampling.

        - **seed** (int) --
          Set a fixed seed for the sampling.

    .. note::
        Precedence of seeding for ``seed_simulator`` is as follows:

        1. ``seed_simulator`` in runtime (i.e. in :meth:`__call__`)
        2. ``seed`` in runtime (i.e. in :meth:`__call__`)
        3. ``seed_simulator`` of ``backend_options``.
        4. default.

        ``seed`` is also used for sampling from a normal distribution when approximation is True.
    """

    def __init__(
        self,
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        observables: BaseOperator | PauliSumOp | Iterable[BaseOperator | PauliSumOp],
        parameters: Iterable[Iterable[Parameter]] | None = None,
        backend_options: dict | None = None,
        transpile_options: dict | None = None,
        approximation: bool = False,
    ):
        """
        Args:
            circuits: Quantum circuits that represent quantum states.
            observables: Observables.
            parameters: Parameters of quantum circuits, specifying the order in which values
                will be bound. Defaults to ``[circ.parameters for circ in circuits]``
                The indexing is such that ``parameters[i, j]`` is the j-th formal parameter of
                ``circuits[i]``.
            backend_options: Options passed to AerSimulator.
            transpile_options: Options passed to transpile.
            approximation: if True, it calculates expectation values with normal distribution
                approximation.
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        circuits = tuple(init_circuit(circuit) for circuit in circuits)

        if isinstance(observables, (PauliSumOp, BaseOperator)):
            observables = (observables,)
        observables = tuple(
            init_observable(observable).simplify(atol=0) for observable in observables
        )

        super().__init__(
            circuits=circuits,
            observables=observables,
            parameters=parameters,
        )
        self._is_closed = False
        backend_options = {} if backend_options is None else backend_options
        method = (
            "density_matrix"
            if approximation and "noise_model" in backend_options
            else "statevector"
        )
        self._backend = AerSimulator(method=method)
        self._backend.set_options(**backend_options)
        self._transpile_options = Options()
        if transpile_options is not None:
            self._transpile_options.update_options(**transpile_options)
        self.approximation = approximation
        self._transpiled_circuits = {}

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

        seed = run_options.pop("seed", None)
        if seed is not None:
            run_options.setdefault("seed_simulator", seed)

        # Key for cache
        key = (tuple(circuits), tuple(observables), self.approximation)

        experiments = []
        parameter_binds = []
        if self.approximation:
            if key in self._transpiled_circuits:
                experiments = self._transpiled_circuits[key]
                for i, j, value in zip(circuits, observables, parameter_values):
                    self._validate_parameter_length(value, i)
                    observable = self._observables[j]
                    parameter_binds.append({k: [v] for k, v in zip(self._parameters[i], value)})
            else:

                for i, j, value in zip(circuits, observables, parameter_values):
                    self._validate_parameter_length(value, i)
                    circuit = self._circuits[i].copy()
                    observable = self._observables[j]
                    circuit.save_expectation_value_variance(observable, range(circuit.num_qubits))
                    experiments.append(circuit)
                    parameter_binds.append({k: [v] for k, v in zip(self._parameters[i], value)})
                experiments = transpile(
                    experiments, self._backend, **self._transpile_options.__dict__
                )
                self._transpiled_circuits[key] = experiments
            # Transpile and run
            shots = run_options.pop("shots", None)
            result = self._backend.run(
                experiments, parameter_binds=parameter_binds, **run_options
            ).result()

            metadata = [{}] * len(circuits)
            if shots is None:
                expectation_values, _ = [
                    result.data(i)["expectation_value_variance"] for i in range(len(circuits))
                ]
                for i, meta in enumerate(metadata):
                    meta["simulator_metadata"] = result.results[i].metadata
            else:
                expectation_values = []
                if seed is None:
                    rng = np.random.default_rng()
                else:
                    rng = np.random.default_rng(seed)
                for i, meta in enumerate(metadata):
                    expectation_value, variance = result.data(i)["expectation_value_variance"]
                    # Sampling from normal distribution
                    standard_error = np.sqrt(variance / shots)
                    expectation_value_with_error = rng.normal(expectation_value, standard_error)
                    expectation_values.append(expectation_value_with_error)
                    meta["variance"] = variance
                    meta["shots"] = shots
                    meta["simulator_metadata"] = result.results[i].metadata
        else:
            num_observable = []
            circuit_data = []

            # Transpile and run
            if key in self._transpiled_circuits:
                experiments, num_observable, circuit_data = self._transpiled_circuits[key]
                for i, j, value in zip(circuits, observables, parameter_values):
                    self._validate_parameter_length(value, i)
                    observable = self._observables[j]
                    for _ in range(len(observable)):
                        parameter_binds.append({k: [v] for k, v in zip(self._parameters[i], value)})
            else:
                for i, j, value in zip(circuits, observables, parameter_values):
                    self._validate_parameter_length(value, i)
                    observable = self._observables[j]
                    num_observable.append(len(observable))
                    circuit = self._circuits[i].copy()
                    num_qubits = circuit.num_qubits
                    # Measurement circuit
                    for pauli, coeff in zip(observable.paulis, observable.coeffs):
                        coeff = np.real_if_close(coeff).item()
                        is_identity = not pauli.x.any() and not pauli.z.any()
                        circuit_data.append({"is_identity": is_identity, "coeff": coeff})
                        # If observable is constant multiplicaition of I, empty circuit is set.
                        if is_identity:
                            experiments.append(QuantumCircuit(0))
                            parameter_binds.append({})
                            continue
                        experiment = circuit.copy()
                        meas_circuit = _measurement_circuit(num_qubits, pauli)
                        for creg in meas_circuit.cregs:
                            experiment.add_register(creg)
                        experiment.compose(meas_circuit, inplace=True)
                        experiments.append(experiment)
                        parameter_binds.append({k: [v] for k, v in zip(self._parameters[i], value)})
                    experiments = transpile(
                        experiments, self._backend, **self._transpile_options.__dict__
                    )
                    self._transpiled_circuits[key] = (experiments, num_observable, circuit_data)
            result = self._backend.run(
                experiments, parameter_binds=parameter_binds, **run_options
            ).result()
            results = result.results
            experiment_index = [0] + list(accumulate(num_observable))

            # Initialize metadata
            metadata = [{}] * len(circuits)

            expectation_values = []
            for start, end, meta in zip(experiment_index, experiment_index[1:], metadata):
                # Initialize
                combined_expval = 0.0
                combined_var = 0.0
                meta["shots"] = 0
                meta["simulator_metadata"] = []
                for result, circuit_datum in zip(results[start:end], circuit_data[start:end]):
                    # If observable is constant multiplicaition of I, expval is trivial.
                    if circuit_datum["is_identity"]:
                        expval, var = 1, 0
                    else:
                        count = result.data.counts
                        meta["simulator_metadata"].append(result.metadata)
                        shots = sum(count.values())
                        meta["shots"] += shots
                        expval, var = _expval_with_variance(count)
                    # Accumulate
                    coeff = circuit_datum["coeff"]
                    combined_expval += expval * coeff
                    combined_var += var * coeff**2
                expectation_values.append(combined_expval)
                meta["variance"] = combined_var

        return EstimatorResult(np.real_if_close(expectation_values), metadata)

    def close(self):
        self._is_closed = True

    def _validate_parameter_length(self, parameter, circuit_index):
        if len(parameter) != len(self._parameters[circuit_index]):
            raise QiskitError(
                f"The number of values ({len(parameter)}) does not match "
                f"the number of parameters ({len(self._parameters[circuit_index])})."
            )


def _measurement_circuit(num_qubits: int, pauli: Pauli):
    qubit_indices = np.arange(pauli.num_qubits)[pauli.z | pauli.x]
    meas_circuit = QuantumCircuit(num_qubits, len(qubit_indices))
    for clbit, i in enumerate(qubit_indices):
        if pauli.x[i]:
            if pauli.z[i]:
                meas_circuit.sdg(i)
            meas_circuit.h(i)
        meas_circuit.measure(i, clbit)
    return meas_circuit


def _expval_with_variance(counts) -> tuple[float, float]:

    denom = 0
    expval = 0
    for bin_outcome, freq in counts.items():
        outcome = int(bin_outcome, 16)
        denom += freq
        expval += freq * (-1) ** bin(outcome).count("1")

    # Divide by total shots
    expval /= denom

    # Compute variance
    variance = 1 - expval**2
    return expval, variance
