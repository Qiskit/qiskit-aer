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
import configparser as cp
import os
from socket import gethostname
import time
import numpy as np

from qiskit import transpile, assemble, execute, user_config, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info import random_unitary
from .aererror import AerError


def profile_performance_options(min_qubits=10, max_qubits=25, ntrials=10,
                                circuit=None, persist=True, gpu=False):
    """Set optimal OpenMP and fusion options for backend."""
    # Profile
    profile = {}

    # Profile OpenMP threshold
    parallel_threshold = None
    try:
        parallel_threshold = profile_parallel_threshold(
            min_qubits=min_qubits, max_qubits=max_qubits, circuit=circuit, ntrials=ntrials)
        profile['statevector_parallel_threshold'] = parallel_threshold
        _PerformanceOptions._set_option('statevector_parallel_threshold',
                                        parallel_threshold, persist)
    except AerError:
        pass

    # Profile CPU and GPU fusion threshold
    for with_gpu in (False, True):
        postfix = '_gpu' if with_gpu else ''
        if with_gpu and not gpu:
            continue
        try:
            qubit_to_costs = profile_fusion_costs(min_qubits=min_qubits,
                                                  max_qubits=max_qubits,
                                                  ntrials=ntrials,
                                                  gpu=with_gpu,
                                                  diagonal=False)
            for num_qubits in qubit_to_costs:
                costs = qubit_to_costs[num_qubits]
                for i, _ in enumerate(costs):
                    profile[f'fusion_cost{postfix}.{num_qubits}.{i + 1}'] = costs[i]
                    _PerformanceOptions._set_option(f'fusion_cost{postfix}.{num_qubits}.{i + 1}',
                                                    costs[i], persist)

            fusion_threshold = profile_fusion_threshold(gpu=with_gpu,
                                                        min_qubits=min_qubits,
                                                        max_qubits=max_qubits,
                                                        ntrials=ntrials,
                                                        circuit=circuit)
            profile[f'fusion_threshold{postfix}'] = fusion_threshold
            _PerformanceOptions._set_option(f'fusion_threshold{postfix}',
                                            fusion_threshold, persist)
        except AerError:
            pass

    return profile


def get_performance_options(num_qubits, gpu=False):
    """Return optimal OpenMP and fusion options for backend."""
    return _PerformanceOptions._get_performance_options(num_qubits, gpu)


def clear_performance_options(persist=True):
    """Clear profiled options for backend."""
    return _PerformanceOptions._clear_performance_options(persist)


class _PerformanceOptions:

    _PERFORMANCE_OPTIONS = {
        'statevector_parallel_threshold': None,
        **{f'fusion_threshold{postfix}': None for postfix in ['', '_gpu']},
        **{f'fusion_cost{postfix}.{num_qubits}.{i + 1}': None
           for num_qubits in range(1, 64)
           for postfix in ['', '_gpu']
           for i in range(5)}
    }

    _DEFAULT_PARALLEL_THRESHOLD = 14
    _DEFAULT_FUSION_THRESHOLD = 14

    _CACHED_PERFORMANCE_OPTIONS = {}

    @staticmethod
    def _get_section_name():
        return f'perf:{gethostname()}'

    @staticmethod
    def _set_option(option_name, value, persist):
        _PerformanceOptions._CACHED_PERFORMANCE_OPTIONS.clear()
        _PerformanceOptions._PERFORMANCE_OPTIONS[option_name] = value
        if persist:
            config = cp.ConfigParser()
            section_name = _PerformanceOptions._get_section_name()
            filename = os.getenv('QISKIT_SETTINGS', user_config.DEFAULT_FILENAME)
            if os.path.isfile(filename):
                config.read(filename)
                if section_name not in config.sections():
                    config.add_section(section_name)
                config.set(section_name, option_name, str(value))
                with open(filename, "w+") as config_file:
                    config.write(config_file)

    @staticmethod
    def _get_options_from_file(option_to_type):
        filename = os.getenv('QISKIT_SETTINGS', user_config.DEFAULT_FILENAME)
        if not os.path.isfile(filename):
            return {}

        config = cp.ConfigParser()
        config.read(filename)

        section_name = _PerformanceOptions._get_section_name()
        if section_name not in config.sections():
            return {}

        ret = {}
        for option_name in config[section_name]:
            ret[option_name] = option_to_type[option_name](config[section_name][option_name])

        return ret

    @staticmethod
    def _get_performance_options(num_qubits, gpu=False):
        """Return optimal OpenMP and fusion options for backend."""

        cache_key = (num_qubits, gpu)
        if cache_key in _PerformanceOptions._CACHED_PERFORMANCE_OPTIONS:
            return _PerformanceOptions._CACHED_PERFORMANCE_OPTIONS[cache_key]

        # Profile
        postfix = '_gpu' if gpu else ''
        option_to_type = {
            'statevector_parallel_threshold': int,
            f'fusion_threshold{postfix}': int,
            **{f'fusion_cost{postfix}.{profiled_qubits}.{i + 1}': float
               for i in range(5)
               for profiled_qubits in range(64)}
        }
        all_profile = _PerformanceOptions._get_options_from_file(option_to_type)

        for option_name in option_to_type:
            if option_name not in _PerformanceOptions._PERFORMANCE_OPTIONS:
                continue
            v = _PerformanceOptions._PERFORMANCE_OPTIONS[option_name]
            if v is not None:
                all_profile[option_name] = v

        ret = {}
        disable_fusion = False
        if 'statevector_parallel_threshold' in all_profile:
            profiled_value = all_profile['statevector_parallel_threshold']
            if num_qubits in range(profiled_value + 1,
                                   _PerformanceOptions._DEFAULT_PARALLEL_THRESHOLD + 1):
                ret['statevector_parallel_threshold'] = profiled_value
        if f'fusion_threshold{postfix}' in all_profile:
            profiled_value = all_profile[f'fusion_threshold{postfix}']
            if num_qubits in range(profiled_value + 1,
                                   _PerformanceOptions._DEFAULT_FUSION_THRESHOLD + 1):
                ret['fusion_threshold'] = profiled_value
            disable_fusion = num_qubits < profiled_value

        if not disable_fusion:
            for profiled_qubits in range(1, num_qubits + 1):
                for i in range(5):
                    profile_name = f'fusion_cost{postfix}.{profiled_qubits}.{i + 1}'
                    if profile_name in all_profile:
                        ret[f'fusion_cost.{i + 1}'] = all_profile[profile_name]

        _PerformanceOptions._CACHED_PERFORMANCE_OPTIONS[cache_key] = ret
        return ret

    @staticmethod
    def _clear_performance_options(persist=True):
        """clear profiled options for backend."""
        _PerformanceOptions._PERFORMANCE_OPTIONS = {
            'statevector_parallel_threshold': None,
            **{f'fusion_threshold{postfix}': None for postfix in ['', '_gpu']},
            **{f'fusion_cost{postfix}.{num_qubits}.{i + 1}': None
               for num_qubits in range(1, 64)
               for postfix in ['', '_gpu']
               for i in range(5)}
        }

        _PerformanceOptions._CACHED_PERFORMANCE_OPTIONS.clear()

        if persist:
            config = cp.ConfigParser()
            filename = os.getenv('QISKIT_SETTINGS', user_config.DEFAULT_FILENAME)
            if os.path.isfile(filename):
                config.read(filename)
                section_name = _PerformanceOptions._get_section_name()
                if section_name in config.sections():
                    config.remove_section(section_name)
                with open(filename, "w+") as config_file:
                    config.write(config_file)


def _generate_profile_circuit(profile_qubit, base_circuit=None, basis_gates=None):
    if profile_qubit < 3:
        raise AerError(f'number of qubit is too small: {profile_qubit}')

    if base_circuit is None:
        return transpile(QuantumVolume(profile_qubit), basis_gates=['id', 'u', 'cx'])

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


def _profile_run(simulator, ntrials, backend_options, qubit, circuit, 
                 basis_gates=None, use_time_taken=False, return_all=False):

    profile_circuit = _generate_profile_circuit(qubit, circuit, basis_gates)

    qobj = assemble(profile_circuit, shots=1)

    total_time_taken = 0.0
    time_taken_list = []
    total_time_elapsed = 0.0
    time_elapsed_list = []
    for _ in range(ntrials):
        start_ts = time.time()
        result = simulator.run(qobj, **backend_options).result()
        end_ts = time.time()

        if not result.success:
            raise AerError('Failed to run a profile circuit')

        total_time_taken += result.results[0].time_taken
        time_taken_list.append(result.results[0].time_taken)
        total_time_elapsed += (end_ts - start_ts)
        time_elapsed_list.append(end_ts - start_ts)

    if use_time_taken:
        if return_all:
            return time_taken_list
        else:
            return total_time_taken
    else:
        if return_all:
            return time_elapsed_list
        else:
            return total_time_elapsed


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

    # pylint: disable=C0415
    from .backends.qasm_simulator import QasmSimulator
    simulator = QasmSimulator()
    basis_gates = ['id', 'u', 'cx']

    ratios = []
    for qubit in range(min_qubits, max_qubits + 1):
        profile_opts['statevector_parallel_threshold'] = 64
        serial_time_taken = _profile_run(simulator, ntrials, profile_opts,
                                         qubit, circuit, basis_gates, return_all=True)
        profile_opts['statevector_parallel_threshold'] = 1
        parallel_time_taken = _profile_run(simulator, ntrials, profile_opts,
                                           qubit, circuit, basis_gates, return_all=True)
        if return_ratios:
            ratios.append(sum(serial_time_taken) / sum(parallel_time_taken))
            break
        if max(parallel_time_taken) >= min(serial_time_taken):
            continue
        if not return_ratios:
            return qubit - 1

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

    # pylint: disable=C0415
    from .backends.qasm_simulator import QasmSimulator
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
                                             qubit, circuit, basis_gates, return_all=True)
        profile_opts['fusion_enable'] = True
        fusion_time_taken = _profile_run(simulator, ntrials, profile_opts,
                                         qubit, circuit, basis_gates, return_all=True)
        if return_ratios:
            ratios.append(sum(non_fusion_time_taken) / sum(fusion_time_taken))
            break
        if max(fusion_time_taken) >= min(non_fusion_time_taken):
            continue
        if not return_ratios:
            return qubit - 1

    if return_ratios:
        return ratios

    raise AerError(f'Unable to find threshold in range [{min_qubits}, {max_qubits}]')


def profile_fusion_costs(min_qubits=10, max_qubits=20, ntrials=10, backend_options=None,
                         gpu=False, diagonal=False, return_ratio=True):
    """Evaluate optimal costs in cost-based fusion for current system."""
    profile_opts = {'method': 'statevector',
                    'max_parallel_experiments': 1,
                    'max_parallel_shots': 1,
                    'fusion_enable': False}

    if backend_options is not None:
        for key, val in backend_options.items():
            profile_opts[key] = val

    # pylint: disable=C0415
    from .backends.qasm_simulator import QasmSimulator
    simulator = QasmSimulator()

    # Ensure fusion is enabled and method is statevector
    if gpu:
        # Check GPU is supported
        result = execute(QuantumVolume(1), simulator, method='statevector_gpu').result()
        if not result.success:
            raise AerError('"statevector_gpu" backend is not supported on this system.')
        profile_opts['method'] = 'statevector_gpu'

    qubit_to_costs = {}
    for num_qubits in range(min_qubits, max_qubits + 1):
        all_gate_time = []
        for target in range(5, 0, -1):
            # Generate a circuit that consists of only unitary/diagonal gates with target-qubit
            profile_circuit = QuantumCircuit(num_qubits)
            loop = 32
            if num_qubits > 20:
                loop = max(1, int(loop / (2 ** (num_qubits - 20))))
            if diagonal:
                for i in range(0, loop):
                    qubits = [q % num_qubits for q in range(i, i + target)]
                    profile_circuit.diagonal([1, -1] * (2 ** (target + 1), qubits))
            else:
                for i in range(0, loop):
                    qubits = [q % num_qubits for q in range(i, i + target)]
                    profile_circuit.unitary(random_unitary(2 ** target), qubits)

            all_gate_time.append(_profile_run(simulator,
                                              ntrials,
                                              profile_opts,
                                              num_qubits,
                                              profile_circuit,
                                              basis_gates=None,
                                              use_time_taken=True))
        all_gate_time.reverse()
        base_line = all_gate_time[0] if return_ratio else 1.0
        costs = [gate_time / base_line for gate_time in all_gate_time]
        qubit_to_costs[num_qubits] = costs

    return qubit_to_costs
