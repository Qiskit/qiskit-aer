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

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.primitives import EstimatorResult
from qiskit.primitives.utils import init_circuit, init_observable
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .. import AerSimulator
from .base_estimator import BaseEstimator  # TODO: fix import path after Terra 0.21.


class Estimator(BaseEstimator):
    """
    Aer implmentation of Estimator.

    :Run Options:
        - **shots** (None or int) --
          The number of shots. If None, it calculates the exact expectation
          values. Otherwise, it samples from normal distributions with standard errors as standard
          deviations using normal distribution approximation.

        - **seed_primitive** (np.random.Generator or int) --
          Set a fixed seed or generator for rng. If shots is None, this option is ignored.
    """

    def __init__(
        self,
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        observables: BaseOperator | PauliSumOp | Iterable[BaseOperator | PauliSumOp],
        parameters: Iterable[Iterable[Parameter]] | None = None,
        backend_options: dict | None = None,
        transpile_options: dict | None = None,
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
        self._backend = AerSimulator()
        backend_options = {} if backend_options is None else backend_options
        self._backend.set_options(**backend_options)
        self._transpile_options = {} if transpile_options is None else transpile_options

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        if self._is_closed:
            raise QiskitError("The primitive has been closed.")

        shots = run_options.pop("shots", None)
        seed_primitive = run_options.pop("seed_primitive", None)
        if seed_primitive is None:
            rng = np.random.default_rng()
        elif isinstance(seed_primitive, np.random.Generator):
            rng = seed_primitive
        else:
            rng = np.random.default_rng(seed_primitive)

        experiments = []
        parameter_binds = []
        for i, j, value in zip(circuits, observables, parameter_values):
            if len(value) != len(self._parameters[i]):
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(self._parameters[i])})."
                )

            circuit = self._circuits[i]
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
        experiments = transpile(experiments, self._backend, **self._transpile_options)
        result = self._backend.run(
            experiments, parameter_binds=parameter_binds, **run_options
        ).result()

        # Initialize metadata
        metadata = [{}] * len(circuits)

        if shots is None:
            expectation_values = [result.data(i)["expectation_value"] for i in range(len(metadata))]
        else:
            expectation_values = []
            for i, meta in enumerate(metadata):
                expectation_value, variance = result.data(i)["expectation_value_variance"]
                standard_error = np.sqrt(variance / shots)
                expectation_value_with_error = rng.normal(expectation_value, standard_error)
                expectation_values.append(expectation_value_with_error)
                meta["variance"] = variance
                meta["shots"] = shots

        return EstimatorResult(np.real_if_close(expectation_values), metadata)

    def close(self):
        self._is_closed = True
