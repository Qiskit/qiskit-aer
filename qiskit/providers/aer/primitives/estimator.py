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

import copy
from collections.abc import Iterable, Sequence
from itertools import accumulate

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.primitives.utils import init_circuit, init_observable
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result.mitigation.utils import str2diag

from qiskit.providers import Options

from .. import AerSimulator


class Estimator(BaseEstimator):
    """
    Aer implmentation of Estimator.

    :Run Options:
        - **shots** (None or int) --
          The number of shots. If None and approximation is True, it calculates the exact
          expectation values. Otherwise, it calculates expectation values with sampling.

        - **seed** (np.random.Generator or int) --
          Set a fixed seed or generator for the normal distribution. If shots is None,
          this option is ignored.
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
        observables = tuple(init_observable(observable) for observable in observables)

        super().__init__(
            circuits=circuits,
            observables=observables,
            parameters=parameters,
        )
        self._is_closed = False
        backend_options = {} if backend_options is None else backend_options
        method = "density_matrix" if "noise_model" in backend_options else "statevector"
        self._backend = AerSimulator(method=method)
        self._backend.set_options(**backend_options)
        self._transpile_options = Options()
        if transpile_options is not None:
            self._transpile_options.update_options(**transpile_options)
        self.approximation = approximation

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

        if self.approximation:
            shots = run_options.pop("shots", None)
            if seed is None:
                rng = np.random.default_rng()
            elif isinstance(seed, np.random.Generator):
                rng = seed
            else:
                rng = np.random.default_rng(seed)

            experiments = []
            parameter_binds = []
            for i, j, value in zip(circuits, observables, parameter_values):
                if len(value) != len(self._parameters[i]):
                    raise QiskitError(
                        f"The number of values ({len(value)}) does not match "
                        f"the number of parameters ({len(self._parameters[i])})."
                    )

                circuit = self._circuits[i].copy()
                observable = self._observables[j]
                if circuit.num_qubits != observable.num_qubits:
                    raise QiskitError(
                        f"The number of qubits of a circuit ({circuit.num_qubits}) does not match "
                        f"the number of qubits of a observable ({observable.num_qubits})."
                    )
                if shots is None:
                    circuit.save_expectation_value(observable, range(circuit.num_qubits))
                else:
                    circuit.save_expectation_value_variance(observable, range(circuit.num_qubits))
                experiments.append(circuit)
                parameter = {k: [v] for k, v in zip(self._parameters[i], value)}
                parameter_binds.append(parameter)
            experiments = transpile(experiments, self._backend, **self._transpile_options.__dict__)
            result = self._backend.run(
                experiments, parameter_binds=parameter_binds, **run_options
            ).result()

            # Initialize metadata
            metadata = [{}] * len(circuits)

            if shots is None:
                expectation_values = [
                    result.data(i)["expectation_value"] for i in range(len(circuits))
                ]
                for i, meta in enumerate(metadata):
                    meta["simulator_metadata"] = result.results[i].metadata
            else:
                expectation_values = []
                for i, meta in enumerate(metadata):
                    expectation_value, variance = result.data(i)["expectation_value_variance"]
                    standard_error = np.sqrt(variance / shots)
                    expectation_value_with_error = rng.normal(expectation_value, standard_error)
                    expectation_values.append(expectation_value_with_error)
                    meta["variance"] = variance
                    meta["shots"] = shots
                    meta["simulator_metadata"] = result.results[i].metadata
        else:
            experiments = []
            parameter_binds = []
            num_observable = []
            for i, j, value in zip(circuits, observables, parameter_values):
                if len(value) != len(self._parameters[i]):
                    raise QiskitError(
                        f"The number of values ({len(value)}) does not match "
                        f"the number of parameters ({len(self._parameters[i])})."
                    )

                observable = self._observables[j]
                num_observable.append(len(observable))
                circuit = self._circuits[i].copy()
                num_qubits = circuit.num_qubits
                meas_circuits = []
                # Measurement circuit
                for pauli, coeff in observable.label_iter():
                    meas_circuit = _create_measurement_circuit(num_qubits, pauli)
                    basis = "".join(char for char in pauli if char != "I") or "I"
                    coeff = np.real_if_close(coeff).item()
                    meas_circuit.metadata = {"basis": basis, "coeff": {basis: coeff}}
                    experiment = circuit.copy()
                    for creg in meas_circuit.cregs:
                        experiment.add_register(creg)
                    experiment.compose(meas_circuit, inplace=True)
                    experiment.metadata = meas_circuit.metadata
                    experiments.append(experiment)
                    parameter = {k: [v] for k, v in zip(self._parameters[i], value)}
                    parameter_binds.append(parameter)
            experiments = transpile(experiments, self._backend, **self._transpile_options.__dict__)
            result = self._backend.run(
                experiments, parameter_binds=parameter_binds, seed_simulator=seed, **run_options
            ).result()
            results = result.results
            header_metadata = [res.header.metadata for res in result.results]

            # Initialize metadata
            metadata = [{}] * len(circuits)
            accum = [0] + list(accumulate(num_observable))

            expectation_values = []
            for start, end, meta in zip(accum, accum[1:], metadata):
                combined_expval = 0.0
                combined_var = 0.0
                # combined_stderr = 0.0
                meta["shots"] = 0
                meta["simulator_metadata"] = []
                for result, sim_meta in zip(results[start:end], header_metadata[start:end]):
                    count = result.data.counts
                    meta["simulator_metadata"].append(result.metadata)
                    basis = sim_meta.get("basis", None)
                    coeff = sim_meta.get("coeff", 1)
                    basis_coeff = coeff if isinstance(coeff, dict) else {basis: coeff}
                    for basis, coeff in basis_coeff.items():
                        diagonal = str2diag(basis.translate(_trans)) if basis is not None else None
                        shots = sum(count.values())
                        meta["shots"] += shots
                        # Compute expval component
                        expval, var = _expval_with_variance(count, diagonal)
                        # Accumulate
                        combined_expval += expval * coeff
                        combined_var += var * coeff**2
                        # combined_stderr += np.sqrt(max(var * coeff**2 / shots, 0.0))
                expectation_values.append(combined_expval)
                meta["variance"] = combined_var

        return EstimatorResult(np.real_if_close(expectation_values), metadata)

    def close(self):
        self._is_closed = True


def _create_measurement_circuit(num_qubits: int, pauli: str):
    reversed_pauli = pauli[::-1]
    qubit_indices = [i for i, char in enumerate(reversed_pauli) if char != "I"] or [0]
    meas_circuit = QuantumCircuit(num_qubits, len(qubit_indices))
    for clbit, i in enumerate(qubit_indices):
        val = reversed_pauli[i]
        if val == "Y":
            meas_circuit.sdg(i)
        if val in ["Y", "X"]:
            meas_circuit.h(i)
        meas_circuit.measure(i, clbit)
    return meas_circuit


_trans = str.maketrans({"X": "Z", "Y": "Z"})


def _expval_with_variance(
    counts: Counts,
    diagonal: np.ndarray | None = None,
) -> tuple[float, float]:

    probs = np.fromiter(counts.values(), dtype=float)
    shots = probs.sum()
    probs = probs / shots

    # Get diagonal operator coefficients
    if diagonal is None:
        coeffs = np.array([(-1) ** key.count("1") for key in counts.keys()], dtype=probs.dtype)
    else:
        keys = [int(key, 16) for key in counts.keys()]
        coeffs = np.asarray(diagonal[keys], dtype=probs.dtype)

    # Compute expval
    expval = coeffs.dot(probs)

    # Compute variance
    if diagonal is None:
        # The square of the parity diagonal is the all 1 vector
        sq_expval = np.sum(probs)
    else:
        sq_expval = (coeffs**2).dot(probs)
    variance = sq_expval - expval**2

    # Compute standard deviation
    if variance < 0:
        if not np.isclose(variance, 0):
            warn(
                "Encountered a negative variance in expectation value calculation."
                f"({variance}). Setting standard deviation of result to 0.",
            )
        variance = np.float64(0.0)
    return expval.item(), variance.item()
