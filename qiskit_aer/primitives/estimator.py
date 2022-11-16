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
from copy import copy
from itertools import accumulate

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.primitives.utils import init_circuit, init_observable
from qiskit.providers import Options
from qiskit.quantum_info import Pauli
from qiskit.quantum_info.operators.base_operator import BaseOperator

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
        circuits: QuantumCircuit | Iterable[QuantumCircuit] | None = None,
        observables: BaseOperator | PauliSumOp | Iterable[BaseOperator | PauliSumOp] | None = None,
        parameters: Iterable[Iterable[Parameter]] | None = None,
        backend_options: dict | None = None,
        transpile_options: dict | None = None,
        run_options: dict | None = None,
        approximation: bool = False,
        skip_transpilation: bool = False,
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
            run_options: Options passed to run.
            approximation: If True, it calculates expectation values with normal distribution
                approximation.
            skip_transpilation: If True, transpilation is skipped.
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        if circuits is not None:
            circuits = tuple(init_circuit(circuit) for circuit in circuits)

        if isinstance(observables, (PauliSumOp, BaseOperator)):
            observables = (observables,)
        if observables is not None:
            observables = tuple(
                init_observable(observable).simplify(atol=0) for observable in observables
            )

        super().__init__(
            circuits=circuits,
            observables=observables,
            parameters=parameters,
            options=run_options,
        )
        self._is_closed = False
        backend_options = {} if backend_options is None else backend_options
        method = (
            "density_matrix" if approximation and "noise_model" in backend_options else "automatic"
        )
        self._backend = AerSimulator(method=method)
        self._backend.set_options(**backend_options)
        self._transpile_options = Options()
        if transpile_options is not None:
            self._transpile_options.update_options(**transpile_options)
        self.approximation = approximation
        self._skip_transpilation = skip_transpilation
        self._cache = {}
        self._transpiled_circuits = {}
        self._layouts = {}

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

        if self.approximation:
            return self._compute_with_approximation(
                circuits, observables, parameter_values, run_options, seed
            )
        else:
            return self._compute(circuits, observables, parameter_values, run_options)

    # This method will be used after Terra 0.22.
    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> PrimitiveJob:
        # pylint: disable=no-name-in-module, import-error, import-outside-toplevel, no-member
        from qiskit.primitives.primitive_job import PrimitiveJob
        from qiskit.primitives.utils import _circuit_key, _observable_key

        circuit_indices: list = []
        for circuit in circuits:
            index = self._circuit_ids.get(_circuit_key(circuit))
            if index is not None:
                circuit_indices.append(index)
            else:
                circuit_indices.append(len(self._circuits))
                self._circuit_ids[_circuit_key(circuit)] = len(self._circuits)
                self._circuits.append(circuit)
                self._parameters.append(circuit.parameters)
        observable_indices: list = []
        for observable in observables:
            observable = init_observable(observable)
            index = self._observable_ids.get(_observable_key(observable))
            if index is not None:
                observable_indices.append(index)
            else:
                observable_indices.append(len(self._observables))
                self._observable_ids[_observable_key(observable)] = len(self._observables)
                self._observables.append(observable)
        job = PrimitiveJob(
            self._call, circuit_indices, observable_indices, parameter_values, **run_options
        )
        job.submit()
        return job

    def close(self):
        self._is_closed = True

    def _compute(self, circuits, observables, parameter_values, run_options):
        # Key for cache
        key = (tuple(circuits), tuple(observables), self.approximation)
        parameter_binds = []

        # Create expectation value experiments.
        if key in self._cache:  # Use a cache
            experiments, num_observable, experiment_data = self._cache[key]
            for i, j, value in zip(circuits, observables, parameter_values):
                self._validate_parameter_length(value, i)
                observable = self._observables[j]
                for _ in range(len(observable)):
                    parameter_binds.append({k: [v] for k, v in zip(self._parameters[i], value)})
        else:
            experiments = []
            experiment_data = []
            num_observable = []
            self._transpile_circuits(circuits)
            for i, j, value in zip(circuits, observables, parameter_values):
                self._validate_parameter_length(value, i)
                observable = self._observables[j]
                num_observable.append(len(observable))
                circuit = (
                    self._circuits[i] if self._skip_transpilation else self._transpiled_circuits[i]
                )
                # Measurement circuit
                for pauli, coeff in zip(observable.paulis, observable.coeffs):
                    coeff = np.real_if_close(coeff).item()
                    is_identity = not pauli.x.any() and not pauli.z.any()
                    experiment_data.append({"is_identity": is_identity, "coeff": coeff})
                    # If observable is constant multiplicaition of I, empty circuit is set.
                    if is_identity:
                        experiments.append(QuantumCircuit(0))
                        parameter_binds.append({})
                        continue
                    experiment = circuit.copy()
                    meas_circuit = self._measurement_circuit(observable.num_qubits, pauli, i)
                    for creg in meas_circuit.cregs:
                        experiment.add_register(creg)
                    experiment.compose(meas_circuit, inplace=True)
                    experiments.append(experiment)
                    parameter_binds.append({k: [v] for k, v in zip(self._parameters[i], value)})
                self._cache[key] = (experiments, num_observable, experiment_data)
        # Run experiments
        parameter_binds = parameter_binds if any(parameter_binds) else None
        result = self._backend.run(
            experiments, parameter_binds=parameter_binds, **run_options
        ).result()
        results = result.results

        # Post processing (calculate expectation values)
        experiment_index = [0] + list(accumulate(num_observable))
        expectation_values = []
        metadata = []
        for start, end in zip(experiment_index, experiment_index[1:]):
            combined_expval = 0.0
            combined_var = 0.0
            shots = 0
            simulator_metadata = []
            for result, experiment_datum in zip(results[start:end], experiment_data[start:end]):
                # If observable is constant multiplicaition of I, expval is trivial.
                if experiment_datum["is_identity"]:
                    expval, var = 1, 0
                else:
                    count = result.data.counts
                    shots += sum(count.values())
                    expval, var = _expval_with_variance(count)
                    simulator_metadata.append(result.metadata)
                # Accumulate
                coeff = experiment_datum["coeff"]
                combined_expval += expval * coeff
                combined_var += var * coeff**2
            expectation_values.append(combined_expval)
            metadata.append(
                {
                    "shots": shots,
                    "variance": combined_var,
                    "simulator_metadata": simulator_metadata,
                }
            )

        return EstimatorResult(np.real_if_close(expectation_values), metadata)

    def _compute_with_approximation(
        self, circuits, observables, parameter_values, run_options, seed
    ):
        # Key for cache
        key = (tuple(circuits), tuple(observables), self.approximation)
        parameter_binds = []
        shots = run_options.pop("shots", None)
        # Create expectation value experiments.
        if key in self._cache:  # Use a cache
            experiments, experiment_data = self._cache[key]
            for i, j, value in zip(circuits, observables, parameter_values):
                self._validate_parameter_length(value, i)
                parameter_binds.append({k: [v] for k, v in zip(self._parameters[i], value)})
        else:
            self._transpile_circuits(circuits)
            experiments = []
            experiment_data = []
            for i, j, value in zip(circuits, observables, parameter_values):
                self._validate_parameter_length(value, i)
                circuit = (
                    self._circuits[i].copy()
                    if self._skip_transpilation
                    else self._transpiled_circuits[i].copy()
                )
                observable = self._observables[j]
                experiment_data.append(observable)
                if shots is None:
                    circuit.save_expectation_value(observable, self._layouts[i])
                else:
                    for term_ind, pauli in enumerate(observable.paulis):
                        circuit.save_expectation_value(pauli, self._layouts[i], label=str(term_ind))
                experiments.append(circuit)
                parameter_binds.append({k: [v] for k, v in zip(self._parameters[i], value)})
            self._cache[key] = (experiments, experiment_data)
        parameter_binds = parameter_binds if any(parameter_binds) else None
        result = self._backend.run(
            experiments, parameter_binds=parameter_binds, **run_options
        ).result()

        # Post processing (calculate expectation values)
        if shots is None:
            expectation_values = [result.data(i)["expectation_value"] for i in range(len(circuits))]
            metadata = [
                {"simulator_metadata": result.results[i].metadata} for i in range(len(experiments))
            ]
        else:
            expectation_values = []
            rng = np.random.default_rng(seed)
            metadata = []
            for i in range(len(experiments)):
                combined_expval = 0.0
                combined_var = 0.0
                coeffs = np.real_if_close(experiment_data[i].coeffs)
                for term_ind, expval in result.data(i).items():
                    var = 1 - expval**2
                    coeff = coeffs[int(term_ind)]
                    combined_expval += expval * coeff
                    combined_var += var * coeff**2
                # Sampling from normal distribution
                standard_error = np.sqrt(combined_var / shots)
                expectation_values.append(rng.normal(combined_expval, standard_error))
                metadata.append(
                    {
                        "variance": combined_var,
                        "shots": shots,
                        "simulator_metadata": result.results[i].metadata,
                    }
                )

        return EstimatorResult(np.real_if_close(expectation_values), metadata)

    def _validate_parameter_length(self, parameter, circuit_index):
        if len(parameter) != len(self._parameters[circuit_index]):
            raise ValueError(
                f"The number of values ({len(parameter)}) does not match "
                f"the number of parameters ({len(self._parameters[circuit_index])})."
            )

    def _measurement_circuit(self, num_qubits: int, pauli: Pauli, circuit_index: int):
        qubit_indices = np.arange(pauli.num_qubits)[pauli.z | pauli.x]
        meas_circuit = QuantumCircuit(num_qubits, len(qubit_indices))
        for clbit, i in enumerate(qubit_indices):
            if pauli.x[i]:
                if pauli.z[i]:
                    meas_circuit.sdg(i)
                meas_circuit.h(i)
            meas_circuit.measure(i, clbit)
        if self._skip_transpilation:
            return meas_circuit
        transpile_opts = copy(self._transpile_options)
        transpile_opts.update_options(initial_layout=self._layouts[circuit_index])
        return transpile(meas_circuit, self._backend, **transpile_opts.__dict__)

    def _transpile(self, circuits):
        if self._skip_transpilation:
            return circuits
        return transpile(circuits, self._backend, **self._transpile_options.__dict__)

    def _transpile_circuits(self, circuits):
        if self._skip_transpilation:
            for i in set(circuits):
                num_qubits = self._circuits[i].num_qubits
                self._layouts[i] = list(range(num_qubits))
            return
        for i in set(circuits):
            if i not in self._transpiled_circuits:
                circuit = self._circuits[i].copy()
                circuit.measure_all()
                num_qubits = circuit.num_qubits
                circuit = self._transpile(circuit)
                bit_map = {bit: index for index, bit in enumerate(circuit.qubits)}
                layout = [bit_map[qr[0]] for _, qr, _ in circuit[-num_qubits:]]
                circuit.remove_final_measurements()
                self._transpiled_circuits[i] = circuit
                self._layouts[i] = layout


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
