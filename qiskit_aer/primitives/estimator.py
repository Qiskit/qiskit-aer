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

from collections import defaultdict
from collections.abc import Sequence
from copy import copy

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.primitives.utils import _circuit_key, _observable_key, init_observable
from qiskit.providers import Options
from qiskit.quantum_info import Pauli, PauliList
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result.models import ExperimentResult

from .. import AerError, AerSimulator


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
        *,
        backend_options: dict | None = None,
        transpile_options: dict | None = None,
        run_options: dict | None = None,
        approximation: bool = False,
        skip_transpilation: bool = False,
        abelian_grouping: bool = True,
    ):
        """
        Args:
            backend_options: Options passed to AerSimulator.
            transpile_options: Options passed to transpile.
            run_options: Options passed to run.
            approximation: If True, it calculates expectation values with normal distribution
                approximation.
            skip_transpilation: If True, transpilation is skipped.
            abelian_grouping: Whether the observable should be grouped into commuting
        """
        super().__init__(options=run_options)

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
        self._circuit_ids = {}
        self._observable_ids = {}
        self._abelian_grouping = abelian_grouping

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:

        seed = run_options.pop("seed", None)
        if seed is not None:
            run_options.setdefault("seed_simulator", seed)

        if self.approximation:
            return self._compute_with_approximation(
                circuits, observables, parameter_values, run_options, seed
            )
        else:
            return self._compute(circuits, observables, parameter_values, run_options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> PrimitiveJob:

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

    def _compute(self, circuits, observables, parameter_values, run_options):
        # Key for cache
        key = (tuple(circuits), tuple(observables), self.approximation)

        # Create expectation value experiments.
        if key in self._cache:  # Use a cache
            experiments_dict, obs_maps = self._cache[key]
            param_binds_dict, param_inds = self._pre_process_params(circuits, parameter_values)
            experiments, parameter_binds = self._flatten(experiments_dict, param_binds_dict)
            post_processings = self._create_post_processing(
                circuits, observables, param_inds, obs_maps, param_binds_dict
            )
        else:
            experiments_dict = {}
            obs_maps = {}
            self._transpile_circuits(circuits)
            circ_obs_map = defaultdict(list)
            for circ_ind, obs_ind in zip(circuits, observables):
                circ_obs_map[circ_ind].append(obs_ind)
            for circ_ind, obs_indices in circ_obs_map.items():
                pauli_list = sum(
                    [self._observables[obs_ind].paulis for obs_ind in obs_indices]
                ).unique()
                if self._abelian_grouping:
                    pauli_lists = pauli_list.group_commuting(qubit_wise=True)
                else:
                    pauli_lists = [PauliList(pauli) for pauli in pauli_list]
                obs_map = defaultdict(list)
                for obs_ind in obs_indices:
                    for pauli in self._observables[obs_ind].paulis:
                        for exp_ind, pauli_list in enumerate(pauli_lists):
                            if pauli in pauli_list:
                                obs_map[obs_ind].append(exp_ind)
                                break
                obs_maps[circ_ind] = obs_map
                paulis = [
                    Pauli(
                        (
                            np.logical_or.reduce(pauli_list.z),  # pylint:disable=no-member
                            np.logical_or.reduce(pauli_list.x),  # pylint:disable=no-member
                        )
                    )
                    for pauli_list in pauli_lists
                ]
                if len(paulis) == 1 and not paulis[0].x.any() and not paulis[0].z.any():
                    break
                meas_circuits = [self._create_meas_circuit(basis, circ_ind) for basis in paulis]
                circuit = (
                    self._circuits[circ_ind]
                    if self._skip_transpilation
                    else self._transpiled_circuits[circ_ind]
                )
                experiments_dict[circ_ind] = self._combine_circs(circuit, meas_circuits)
            self._cache[key] = experiments_dict, obs_maps

            param_binds_dict, param_inds = self._pre_process_params(circuits, parameter_values)

            # Flatten
            experiments, parameter_binds = self._flatten(experiments_dict, param_binds_dict)

            # Create PostProcessing
            post_processings = self._create_post_processing(
                circuits, observables, param_inds, obs_maps, param_binds_dict
            )

        # Run experiments
        if experiments:
            results = (
                self._backend.run(
                    circuits=experiments,
                    parameter_binds=parameter_binds if any(parameter_binds) else None,
                    **run_options,
                )
                .result()
                .results
            )
        else:
            results = []

        # Post processing (calculate expectation values)
        expectation_values = []
        metadata = []
        for post_processing in post_processings:
            expval, meta = post_processing.run(results)
            expectation_values.append(expval)
            metadata.append(meta)
        return EstimatorResult(np.real_if_close(expectation_values), metadata)

    def _pre_process_params(self, circuits, parameter_values):
        param_inds = []
        param_binds_dict = {}
        for circ_ind, param_val in zip(circuits, parameter_values):
            self._validate_parameter_length(param_val, circ_ind)
            if circ_ind in param_binds_dict and len(self._parameters[circ_ind]) > 0:
                param_inds.append(len(next(iter(param_binds_dict[circ_ind].values()))))
                for k, v in zip(self._parameters[circ_ind], param_val):
                    param_binds_dict[circ_ind][k].append(v)
            else:
                param_binds_dict[circ_ind] = {
                    k: [v] for k, v in zip(self._parameters[circ_ind], param_val)
                }
                param_inds.append(0)
        return param_binds_dict, param_inds

    def _create_meas_circuit(self, basis: Pauli, circuit_index):
        qargs = np.arange(basis.num_qubits)[basis.z | basis.x]
        meas_circuit = QuantumCircuit(basis.num_qubits, len(qargs))
        for clbit, qarg in enumerate(qargs):
            if basis.x[qarg]:
                if basis.z[qarg]:
                    meas_circuit.sdg(qarg)
                meas_circuit.h(qarg)
            meas_circuit.measure(qarg, clbit)
        meas_circuit.metadata = {"basis": basis}
        if self._skip_transpilation:
            return meas_circuit
        transpile_opts = copy(self._transpile_options)
        transpile_opts.update_options(initial_layout=self._layouts[circuit_index])
        return transpile(meas_circuit, self._backend, **transpile_opts.__dict__)

    @staticmethod
    def _combine_circs(circuit: QuantumCircuit, meas_circuits: list[QuantumCircuit]):
        circs = []
        for meas_circuit in meas_circuits:
            new_circ = circuit.copy()
            for creg in meas_circuit.cregs:
                new_circ.add_register(creg)
            new_circ.compose(meas_circuit, inplace=True)
            _update_metadata(new_circ, meas_circuit.metadata)
            circs.append(new_circ)
        return circs

    @staticmethod
    def _calculate_result_index(
        obs_maps, param_ind, circ_ind, obs_ind, term_ind, param_binds
    ) -> int:
        circ_dup_num = {
            circ_ind: len(next(iter(v.values()))) if v else 1 for circ_ind, v in param_binds.items()
        }
        obs_dup_num = {
            circ_ind: {obs_ind: max(exp_inds) + 1 for obs_ind, exp_inds in obs_map.items()}
            for circ_ind, obs_map in obs_maps.items()
        }
        result_index = 0
        for _circ_ind, obs_num_dict in obs_dup_num.items():
            for _obs_ind, obs_num in obs_num_dict.items():
                if circ_ind == _circ_ind and obs_ind == _obs_ind:
                    obs_offset = obs_maps[circ_ind][obs_ind][term_ind]
                    result_index += obs_offset * circ_dup_num[circ_ind] + param_ind
                    return result_index
                result_index += circ_dup_num[_circ_ind] * obs_num
        raise AerError(
            "Bug. Please report from isssue: https://github.com/Qiskit/qiskit-aer/issues"
        )

    @staticmethod
    def _flatten(experiments, param_binds):
        experiments_list = []
        parameter_binds = []
        for circ_ind in experiments:
            experiments_list.extend(experiments[circ_ind])
            for _ in range(len(experiments[circ_ind])):
                parameter_binds.append(param_binds[circ_ind])
        return experiments_list, parameter_binds

    def _create_post_processing(
        self, circuits, observables, param_inds, obs_maps, param_binds
    ) -> list[_PostProcessing]:
        post_processings = []
        for circ_ind, obs_ind, param_ind in zip(circuits, observables, param_inds):
            result_indices = []
            paulis = []
            coeffs = []
            observable = self._observables[obs_ind]
            for term_ind, (pauli, coeff) in enumerate(zip(observable.paulis, observable.coeffs)):
                # Identity
                if not pauli.x.any() and not pauli.z.any():
                    result_indices.append(None)
                    paulis.append(PauliList(pauli))
                    coeffs.append([coeff])
                    continue

                result_index = self._calculate_result_index(
                    obs_maps, param_ind, circ_ind, obs_ind, term_ind, param_binds
                )
                if result_index in result_indices:
                    i = result_indices.index(result_index)
                    paulis[i] += pauli
                    coeffs[i].append(coeff)
                else:
                    result_indices.append(result_index)
                    paulis.append(PauliList(pauli))
                    coeffs.append([coeff])
            post_processings.append(_PostProcessing(result_indices, paulis, coeffs))
        return post_processings

    def _compute_with_approximation(
        self, circuits, observables, parameter_values, run_options, seed
    ):
        if self._abelian_grouping:
            raise AerError("approximation and abelian_grouping cannot be True at the same time.")
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


class _PostProcessing:
    def __init__(
        self, circuit_indices: list[int], paulis: list[PauliList], coeffs: list[list[float]]
    ):
        self._circuit_indices = circuit_indices
        self._paulis = paulis
        self._coeffs = [np.array(c) for c in coeffs]

    def run(self, results: list[ExperimentResult]) -> tuple[float, dict]:
        """Coumpute expectation value.

        Args:
            results: list of ExperimentResult.

        Returns:
            tuple of an expectation value and metadata.
        """
        combined_expval = 0.0
        combined_var = 0.0
        simulator_metadata = []
        for c_i, paulis, coeffs in zip(self._circuit_indices, self._paulis, self._coeffs):
            if c_i is None:
                # Observable is identity
                expvals, variances = [1], [0]
                shots = 0
            else:
                result = results[c_i]
                count = result.data.counts
                shots = sum(count.values())
                basis = result.header.metadata["basis"]
                indices = np.where(basis.z | basis.x)[0]
                measured_paulis = PauliList.from_symplectic(
                    paulis.z[:, indices], paulis.x[:, indices], 0
                )
                expvals, variances = _pauli_expval_with_variance(count, measured_paulis)
                simulator_metadata.append(result.metadata)
            combined_expval += np.dot(expvals, coeffs)
            combined_var += np.dot(variances, coeffs**2)
        metadata = {
            "shots": shots,
            "variance": combined_var,
            "simulator_metadata": simulator_metadata,
        }
        return combined_expval, metadata


def _update_metadata(circuit: QuantumCircuit, metadata: dict) -> QuantumCircuit:
    if circuit.metadata:
        circuit.metadata.update(metadata)
    else:
        circuit.metadata = metadata
    return circuit


def _pauli_expval_with_variance(counts: dict, paulis: PauliList) -> tuple[np.ndarray, np.ndarray]:
    # Diag indices
    size = len(paulis)
    diag_inds = _paulis2inds(paulis)

    expvals = np.zeros(size, dtype=float)
    denom = 0  # Total shots for counts dict
    for hex_outcome, freq in counts.items():
        outcome = int(hex_outcome, 16)
        denom += freq
        for k in range(size):
            coeff = (-1) ** _parity(diag_inds[k] & outcome)
            expvals[k] += freq * coeff

    # Divide by total shots
    expvals /= denom

    variances = 1 - expvals**2
    return expvals, variances


def _paulis2inds(paulis: PauliList) -> list[int]:
    nonid = paulis.z | paulis.x
    packed_vals = np.packbits(nonid, axis=1, bitorder="little").astype(  # pylint:disable=no-member
        object
    )
    power_uint8 = 1 << (8 * np.arange(packed_vals.shape[1], dtype=object))
    inds = packed_vals @ power_uint8
    return inds.tolist()


def _parity(integer: int) -> int:
    """Return the parity of an integer"""
    return bin(integer).count("1") % 2
