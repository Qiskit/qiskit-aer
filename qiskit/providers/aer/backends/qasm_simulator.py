# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Qiskit Aer qasm simulator backend.
"""

import logging
from math import log2
from qiskit._util import local_hardware_info
from qiskit.providers.models import BackendConfiguration
from .aerbackend import AerBackend
from qasm_controller_wrapper import qasm_controller_execute
from ..aererror import AerError
from ..version import __version__

logger = logging.getLogger(__name__)


class QasmSimulator(AerBackend):
    """Aer quantum circuit simulator

    Backend options:

        The following backend options may be used with in the
        `backend_options` kwarg diction for `QasmSimulator.run` or
        `qiskit.execute`

        * "method" (str): Set the simulation method. Allowed values are:
            * "statevector": Uses a dense statevector simulation.
            * "stabilizer": uses a Clifford stabilizer state simulator that
            is only valid for Clifford circuits and noise models.
            * "automatic": automatically run on stabilizer simulator if
            the circuit and noise model supports it, otherwise use the
            statevector method (Default: "automatic").

        * "initial_statevector" (vector_like): Sets a custom initial
            statevector for the simulation instead of the all zero
            initial state (Default: None).

        * "chop_threshold" (double): Sets the threshold for truncating small
            values to zero in the Result data (Default: 1e-15)

        * "max_parallel_threads" (int): Sets the maximum number of CPU
            cores used by OpenMP for parallelization. If set to 0 the
            maximum will be set to the number of CPU cores (Default: 0).

        * "max_parallel_experiments" (int): Sets the maximum number of
            qobj experiments that may be executed in parallel up to the
            max_parallel_threads value. If set to 1 parallel circuit
            execution will be disabled. If set to 0 the maximum will be
            automatically set to max_parallel_threads (Default: 1).

        * "max_parallel_shots" (int): Sets the maximum number of
            shots that may be executed in parallel during each experiment
            execution, up to the max_parallel_threads value. If set to 1
            parallel shot execution wil be disabled. If set to 0 the
            maximum will be automatically set to max_parallel_threads.
            Note that this cannot be enabled at the same time as parallel
            experiment execution (Default: 1).

        * "statevector_parallel_threshold" (int): Sets the threshold that
            "n_qubits" must be greater than to enable OpenMP
            parallelization for matrix multiplication during execution of
            an experiment. If parallel circuit or shot execution is enabled
            this will only use unallocated CPU cores up to
            max_parallel_threads. Note that setting this too low can reduce
            performance (Default: 12).

        * "statevector_sample_measure_opt" (int): Sets the threshold that
            the number of qubits must be greater than to enable a large
            qubit optimized implementation of measurement sampling. Note
            that setting this two low can reduce performance (Default: 10)

        * "statevector_hpc_gate_opt" (bool): If set to True this enables
            a different optimzied gate application routine that can
            increase performance on systems with a large number of CPU
            cores. For systems with a small number of cores it enabling
            can reduce performance (Default: False).
    """

    MAX_QUBIT_MEMORY = int(log2(local_hardware_info()['memory'] * (1024 ** 3) / 16))

    DEFAULT_CONFIGURATION = {
        'backend_name': 'qasm_simulator',
        'backend_version': __version__,
        'n_qubits': MAX_QUBIT_MEMORY,
        'url': 'https://github.com/Qiskit/qiskit-aer',
        'simulator': True,
        'local': True,
        'conditional': True,
        'open_pulse': False,
        'memory': True,
        'max_shots': 100000,
        'description': 'A C++ simulator with realistic for qobj files',
        'basis_gates': ['u1', 'u2', 'u3', 'cx', 'cz', 'id', 'x', 'y', 'z',
                        'h', 's', 'sdg', 't', 'tdg', 'ccx', 'swap',
                        'snapshot', 'unitary'],
        'gates': [
            {
                'name': 'TODO',
                'parameters': [],
                'qasm_def': 'TODO'
            }
        ]
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(qasm_controller_execute,
                         BackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION),
                         provider=provider)

    def _validate(self, qobj, backend_options, noise_model):
        """Semantic validations of the qobj which cannot be done via schemas.

        1. Check number of qubits will fit in local memory.
        2. warn if no classical registers or measurements in circuit.
        """
        clifford_instructions = ["id", "x", "y", "z", "h", "s", "sdg",
                                 "CX", "cx", "cz", "swap",
                                 "barrier", "reset", "measure"]
        # Check if noise model is Clifford:
        method = "automatic"
        if backend_options and "method" in backend_options:
            method = backend_options["method"]

        clifford_noise = not (method == "statevector")

        if clifford_noise:
            if method != "stabilizer" and noise_model:
                for error in noise_model.as_dict()['errors']:
                    if error['type'] == 'qerror':
                        for circ in error["instructions"]:
                            for instr in circ:
                                if instr not in clifford_instructions:
                                    clifford_noise = False
                                    break
        # Check to see if experiments are clifford
        for experiment in qobj.experiments:
            name = experiment.header.name
            # Check for classical bits
            if experiment.config.memory_slots == 0:
                logger.warning('No classical registers in circuit "%s": '
                               'result data will not contain counts.', name)
            # Check if Clifford circuit or if measure opts missing
            no_measure = True
            clifford = False if method == "statevector" else clifford_noise
            for op in experiment.instructions:
                if not clifford and not no_measure:
                    break  # we don't need to check any more ops
                if clifford and op.name not in clifford_instructions:
                    clifford = False
                if no_measure and op.name == "measure":
                    no_measure = False
            # Print warning if clbits but no measure
            if no_measure:
                logger.warning('No measurements in circuit "%s": '
                               'count data will return all zeros.', name)
            # Check qubits for statevector simulation
            if not clifford:
                n_qubits = experiment.config.n_qubits
                max_qubits = self.configuration().n_qubits
                if n_qubits > max_qubits:
                    system_memory = int(local_hardware_info()['memory'])
                    raise AerError('Number of qubits ({}) '.format(n_qubits) +
                                   'is greater than maximum ({}) '.format(max_qubits) +
                                   'for "{}" (method=statevector) '.format(self.name()) +
                                   'with {} GB system memory.'.format(system_memory))
