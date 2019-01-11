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

    def run(self, qobj, backend_options=None, noise_model=None):
        """Run a qobj on the backend.

        Args:
            qobj (Qobj): a Qobj.
            backend_options (dict): backend configuration options.
            noise_model (NoiseModel): noise model for simulations.

        Returns:
            AerJob: the simulation job.
        """
        return super().run(qobj, backend_options=backend_options,
                           noise_model=noise_model)

    def __init__(self, configuration=None, provider=None):
        super().__init__(qasm_controller_execute,
                         BackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION),
                         provider=provider)

    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas.

        1. Check number of qubits will fit in local memory.
        2. warn if no classical registers or measurements in circuit.
        """
        n_qubits = qobj.config.n_qubits
        max_qubits = self.configuration().n_qubits
        if n_qubits > max_qubits:
            raise AerError('Number of qubits ({}) '.format(n_qubits) +
                           'is greater than maximum ({}) '.format(max_qubits) +
                           'for "{}" '.format(self.name()) +
                           'with {} GB system memory.'.format(int(local_hardware_info()['memory'])))
        for experiment in qobj.experiments:
            name = experiment.header.name
            if experiment.config.memory_slots == 0:
                logger.warning('No classical registers in circuit "%s": '
                               'result data will not contain counts.', name)
            elif 'measure' not in [op.name for op in experiment.instructions]:
                logger.warning('No measurements in circuit "%s": '
                               'count data will return all zeros.', name)
