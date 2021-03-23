# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Profile backend options for optimal performance
"""
import numpy as np

from qiskit import transpile, assemble, execute, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info import random_unitary
from .aererror import AerError
from .backends.aerbackend import AerBackend
from .backends.qasm_simulator import QasmSimulator


def optimize_backend_options(min_qubits=10, max_qubits=25, ntrials=5, circuit=None):
    """Set optimal OpenMP and fusion options for backend."""
    # Profile
    profile = {}

    # Profile OpenMP threshold
    parallel_threshold = None
    try:
        parallel_threshold = profile_parallel_threshold(
            min_qubits=min_qubits, max_qubits=max_qubits, circuit=circuit, ntrials=ntrials)
        profile['statevector_parallel_threshold'] = parallel_threshold
        _set_optimized_option('statevector_parallel_threshold', parallel_threshold)
    except AerError:
        pass

    # Profile CPU and GPU fusion threshold
    for gpu in (False, True):
        postfix = '_gpu' if gpu else ''
        try:
            fusion_threshold = profile_fusion_threshold(gpu=gpu,
                                                        min_qubits=min_qubits,
                                                        max_qubits=max_qubits,
                                                        ntrials=ntrials,
                                                        circuit=circuit)
            profile[f'fusion_threshold{postfix}'] = fusion_threshold
            _set_optimized_option(f'fusion_threshold{postfix}', fusion_threshold)

            default_qubit = 20
            num_qubits = min(max(default_qubit, fusion_threshold), max_qubits)
            costs = profile_fusion_costs(num_qubits,
                                         ntrials=ntrials,
                                         gpu=gpu,
                                         diagonal=False)
            for i, _ in enumerate(costs):
                profile[f'fusion_cost{postfix}.{i + 1}'] = costs[i]
                _set_optimized_option(f'fusion_cost{postfix}.{i + 1}', costs[i])

        except AerError:
            pass

    return profile


def _set_optimized_option(option_name, value):
    setattr(AerBackend, f'_{option_name}', value)


def clear_optimized_backend_options():
    """Set profiled options for backend."""
    _clear_optimized_option('statevector_parallel_threshold')
    for gpu in (False, True):
        postfix = '_gpu' if gpu else ''
        _clear_optimized_option(f'fusion_threshold{postfix}')
        for i in range(5):
            _clear_optimized_option(f'fusion_cost{postfix}.{i + 1}')


def _clear_optimized_option(option_name):
    if hasattr(AerBackend, f'_{option_name}'):
        delattr(AerBackend, f'_{option_name}')


def _generate_profile_circuit(profile_qubit, base_circuit=None, basis_gates=None):
    if profile_qubit < 3:
        raise AerError(f'number of qubit is too small: {profile_qubit}')

    if base_circuit is None:
        return transpile(QuantumVolume(profile_qubit, 10), basis_gates=['id', 'u', 'cx'])

    if basis_gates is not None:
        profile_circuit = transpile(base_circuit.copy(), basis_gates=basis_gates)
    else:
        profile_circuit = base_circuit.copy()

    if profile_qubit < profile_circuit.num_qubits:

        def global_index(qubit):
            ret = 0
            for qreg in profile_circuit.qregs:
                if qreg is qubit.register:
                    return ret + qubit.index
                else:
                    ret += qreg.size
            raise ValueError(f'odd qubit: {qubit}')

        def global_qubit(index):
            orig = index
            for qreg in profile_circuit.qregs:
                if index < qreg.size:
                    return qreg[index]
                else:
                    index -= qreg.size
            raise ValueError(f'odd index: {orig}')

        for i in range(len(profile_circuit.data)):
            inst, qubits, cbits = profile_circuit.data[i]
            new_qubit_idxs = []
            changed = False
            for qubit in qubits:
                gidx = global_index(qubit) % profile_qubit
                while gidx in new_qubit_idxs:
                    gidx = np.random.randint(profile_qubit)
                changed |= (gidx != global_index(qubit))
                new_qubit_idxs.append(gidx)
            if changed:
                profile_circuit.data[i] = (inst,
                                           [global_qubit(idx) for idx in new_qubit_idxs],
                                           cbits)

    elif profile_circuit.num_qubits < profile_qubit:
        # add new qubits
        new_register = QuantumRegister(profile_qubit - profile_circuit.num_qubits)
        profile_circuit.add_register(new_register)
        # add a gate for each new qubit to prevent truncation
        for new_qubit in new_register:
            profile_circuit.u(0, 1, 2, new_qubit)

    return profile_circuit


def _profile_run(simulator, ntrials, backend_options,
                 qubit, circuit, basis_gates=None):

    profile_circuit = _generate_profile_circuit(qubit, circuit, basis_gates)

    qobj = assemble(ntrials * [profile_circuit], shots=1)

    result = simulator.run(qobj, **backend_options).result()

    if not result.success:
        raise AerError('Failed to run a profile circuit')

    time_taken = 0.0
    for j in range(ntrials):
        time_taken += result.results[j].time_taken

    return time_taken


def profile_parallel_threshold(min_qubits=10, max_qubits=20, ntrials=10,
                               circuit=None, backend_options=None, return_ratios=False):
    """Evaluate optimal OMP parallel threshold for current system."""

    profile_opts = {'method': 'statevector',
                    'max_parallel_experiments': 1,
                    'max_parallel_shots': 1,
                    'fusion_enable': False}

    if backend_options is not None:
        for key, val in backend_options.items():
            profile_opts[key] = val

    simulator = QasmSimulator()
    basis_gates = ['id', 'u', 'cx']

    ratios = []
    for qubit in range(min_qubits, max_qubits + 1):
        profile_opts['statevector_parallel_threshold'] = 64
        serial_time_taken = _profile_run(simulator, ntrials, profile_opts,
                                         qubit, circuit, basis_gates)
        profile_opts['statevector_parallel_threshold'] = 1
        parallel_time_taken = _profile_run(simulator, ntrials, profile_opts,
                                           qubit, circuit, basis_gates)
        if return_ratios:
            ratios.append(serial_time_taken / parallel_time_taken)
        elif serial_time_taken < parallel_time_taken:
            return qubit

    if return_ratios:
        return ratios

    raise AerError(f'Unable to find threshold in range [{min_qubits}, {max_qubits}]')


def profile_fusion_threshold(min_qubits=10, max_qubits=20, ntrials=10,
                             circuit=None, backend_options=None, gpu=False,
                             return_ratios=False):
    """Evaluate optimal OMP parallel threshold for current system."""

    profile_opts = {'method': 'statevector',
                    'max_parallel_experiments': 1,
                    'max_parallel_shots': 1,
                    'fusion_threshold': 1}

    if backend_options is not None:
        for key, val in backend_options.items():
            profile_opts[key] = val

    simulator = QasmSimulator()
    basis_gates = ['id', 'u', 'cx']

    # Ensure fusion is enabled and method is statevector
    if gpu:
        # Check GPU is supported
        result = execute(QuantumVolume(1), simulator, method='statevector_gpu').result()
        if not result.success:
            raise AerError('"statevector_gpu" backend is not supported on this system.')
        profile_opts['method'] = 'statevector_gpu'

    ratios = []
    for qubit in range(min_qubits, max_qubits + 1):
        profile_opts['fusion_enable'] = False
        non_fusion_time_taken = _profile_run(simulator, ntrials, profile_opts,
                                             qubit, circuit, basis_gates)
        profile_opts['fusion_enable'] = True
        fusion_time_taken = _profile_run(simulator, ntrials, profile_opts,
                                         qubit, circuit, basis_gates)
        if return_ratios:
            ratios.append(non_fusion_time_taken / fusion_time_taken)
        elif non_fusion_time_taken < fusion_time_taken:
            return qubit

    if return_ratios:
        return ratios

    raise AerError(f'Unable to find threshold in range [{min_qubits}, {max_qubits}]')


def profile_fusion_costs(num_qubits, ntrials=10, backend_options=None, gpu=False, diagonal=False):
    """Evaluate optimal costs in cost-based fusion for current system."""
    profile_opts = {'method': 'statevector',
                    'max_parallel_experiments': 1,
                    'max_parallel_shots': 1,
                    'fusion_enable': False}

    if backend_options is not None:
        for key, val in backend_options.items():
            profile_opts[key] = val

    simulator = QasmSimulator()

    # Ensure fusion is enabled and method is statevector
    if gpu:
        # Check GPU is supported
        result = execute(QuantumVolume(1), simulator, method='statevector_gpu').result()
        if not result.success:
            raise AerError('"statevector_gpu" backend is not supported on this system.')
        profile_opts['method'] = 'statevector_gpu'

    all_gate_time = []
    for target in range(0, 5):
        # Generate a circuit that consists of only unitary/diagonal gates with target-qubit
        profile_circuit = QuantumCircuit(num_qubits)
        if diagonal:
            for i in range(0, 100):
                qubits = [q % num_qubits for q in range(i, i + target + 1)]
                profile_circuit.diagonal([1, -1] * (2 ** target), qubits)
        else:
            for i in range(0, 100):
                qubits = [q % num_qubits for q in range(i, i + target + 1)]
                profile_circuit.unitary(random_unitary(2 ** (target + 1)), qubits)

        all_gate_time.append(_profile_run(simulator,
                                          ntrials,
                                          profile_opts,
                                          num_qubits,
                                          profile_circuit,
                                          None))

    costs = []
    for target in range(0, 5):
        costs.append(all_gate_time[target] / all_gate_time[0])

    return costs
