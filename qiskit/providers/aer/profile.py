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
from qiskit import transpile, assemble, execute
from qiskit.circuit.library import QuantumVolume
from .aererror import AerError
from .backends.aerbackend import AerBackend
from .backends.qasm_simulator import QasmSimulator


def optimize_backend_options(min_qubits=10, max_qubits=20, ntrials=10):
    """Set optimal OpenMP and fusion options for backend."""
    # Profile
    profile = {}

    # Profile OpenMP threshold
    try:
        parallel_threshold = profile_parallel_threshold(
            min_qubits=min_qubits, max_qubits=max_qubits, ntrials=ntrials)

        profile['statevector_parallel_threshold'] = parallel_threshold
    except AerError:
        pass

    # Profile CPU fusion threshold
    try:
        fusion_threshold = profile_fusion_threshold(
            min_qubits=min_qubits, max_qubits=max_qubits, ntrials=ntrials)
        profile['fusion_threshold'] = fusion_threshold
    except AerError:
        pass

    # Profile GPU fusion threshold
    try:
        fusion_threshold_gpu = profile_fusion_threshold(
            gpu=True, min_qubits=min_qubits, max_qubits=max_qubits, ntrials=ntrials)
        profile['fusion_threshold_gpu'] = fusion_threshold_gpu
    except AerError:
        pass

    # TODO: Write profile to a local qiskitaerrc file so this doesn't
    # need to be re-run on a system and the following can be loaded
    # in the AerBackend class from the rc file if it is found
    if 'statevector_parallel_threshold' in profile:
        AerBackend._statevector_parallel_threshold = profile[
            'statevector_parallel_threshold']
    if 'fusion_threshold' in profile:
        AerBackend._fusion_threshold = profile['fusion_threshold']
    if 'fusion_threshold_gpu' in profile:
        AerBackend._fusion_threshold_gpu = profile['fusion_threshold_gpu']

    return profile


def profile_parallel_threshold(min_qubits=10, max_qubits=20, ntrials=10,
                               backend_options=None,
                               return_ratios=False):
    """Evaluate optimal OMP parallel threshold for current system."""
    simulator = QasmSimulator()
    opts = {'method': 'statevector',
            'max_parallel_experiments': 1,
            'max_parallel_shots': 1,
            'fusion_enabled': False}
    if backend_options:
        for key, val in backend_options.items():
            opts[key] = val
    # Ensure method is statevector
    opts['method'] = 'statevector'

    omp_ratios = []
    qubit_threshold = None

    for i in range(min_qubits, max_qubits + 1):
        circ = transpile(QuantumVolume(i, 10),
                         basis_gates=['id', 'u1', 'u2', 'u3', 'cx', 'swap'])

        qobj = assemble(ntrials * [circ], shots=1)
        times = []
        for val in [i - 1, i]:
            opts['statevector_parallel_threshold'] = val
            result = simulator.run(qobj, backend_options=opts).result()
            time_taken = 0.0
            for j in range(ntrials):
                time_taken += result.results[j].time_taken
            times.append(time_taken)

        # Compute ratio
        ratio = times[1] / times[0]
        omp_ratios.append((i, ratio))

        if qubit_threshold is None:
            if ratio > 1:
                qubit_threshold = i
        elif ratio < 1:
            qubit_threshold = None
        else:
            break

    # Check if we found a threshold in the provided range
    if qubit_threshold is None:
        raise AerError('Unable to find threshold in range [{}, {}]'.format(
            min_qubits, max_qubits))

    if return_ratios:
        return qubit_threshold, omp_ratios
    return qubit_threshold


def profile_fusion_threshold(min_qubits=10, max_qubits=20, ntrials=10,
                             backend_options=None, gpu=False,
                             return_ratios=False):
    """Evaluate optimal OMP parallel threshold for current system."""
    simulator = QasmSimulator()
    opts = {'method': 'statevector',
            'max_parallel_experiments': 1,
            'max_parallel_shots': 1}
    if backend_options:
        for key, val in backend_options.items():
            opts[key] = val

    # Ensure fusion is enabled and method is statevector
    if gpu:
        opts['method'] = 'statevector_gpu'
        # Check GPU is supported
        result = execute(QuantumVolume(1), simulator, backend_options=opts).result()
        if not result.success:
            raise AerError('"statevector_gpu" backend is not supported on this system.')
    else:
        opts['method'] = 'statevector'
    opts['fusion_enabled'] = True

    omp_ratios = []
    qubit_threshold = None

    for i in range(min_qubits, max_qubits + 1):
        circ = transpile(QuantumVolume(i, 10),
                         basis_gates=['id', 'u1', 'u2', 'u3', 'cx', 'swap'])

        qobj = assemble(ntrials * [circ], shots=1)
        times = []
        for val in [i, i + 1]:
            opts['fusion_threshold'] = val
            result = simulator.run(qobj, backend_options=opts).result()
            time_taken = 0.0
            for j in range(ntrials):
                time_taken += result.results[j].time_taken
            times.append(time_taken)

        # Compute ratio
        ratio = times[1] / times[0]
        omp_ratios.append((i, ratio))

        if qubit_threshold is None:
            if ratio > 1:
                qubit_threshold = i
        elif ratio < 1:
            qubit_threshold = None
        else:
            break

    # Check if we found a threshold in the provided range
    if qubit_threshold is None:
        raise AerError('Unable to find stable threshold in range [{}, {}]'.format(
            min_qubits, max_qubits))

    if return_ratios:
        return qubit_threshold, omp_ratios
    return qubit_threshold
