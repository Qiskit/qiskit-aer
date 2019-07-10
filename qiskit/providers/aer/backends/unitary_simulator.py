# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Qiskit Aer Unitary Simulator Backend.
"""

import logging
import os
from math import log2, sqrt
from qiskit.util import local_hardware_info
from qiskit.providers.models import BackendConfiguration

from .aerbackend import AerBackend
from ..aererror import AerError
# pylint: disable=import-error
from .unitary_controller_wrapper import unitary_controller_execute
from ..version import __version__

# Logger
logger = logging.getLogger(__name__)


class UnitarySimulator(AerBackend):
    """Unitary circuit simulator.

    Backend options:

        The following backend options may be used with in the
        `backend_options` kwarg diction for `UnitarySimulator.run` or
        `qiskit.execute`

        * "initial_unitary" (matrix_like): Sets a custom initial unitary
            matrix for the simulation instead of identity (Default: None).

        * "zero_threshold" (double): Sets the threshold for truncating
            small values to zero in the result data (Default: 1e-10).

        * "max_parallel_threads" (int): Sets the maximum number of CPU
            cores used by OpenMP for parallelization. If set to 0 the
            maximum will be set to the number of CPU cores (Default: 0).

        * "max_parallel_experiments" (int): Sets the maximum number of
            qobj experiments that may be executed in parallel up to the
            max_parallel_threads value. If set to 1 parallel circuit
            execution will be disabled. If set to 0 the maximum will be
            automatically set to max_parallel_threads (Default: 1).

        * "max_memory_mb" (int): Sets the maximum size of memory
            to store a state vector. If a state vector needs more, an error
            is thrown. In general, a state vector of n-qubits uses 2^n complex
            values (16 Bytes). If set to 0, the maximum will be automatically
            set to half the system memory size (Default: 0).

        * "statevector_parallel_threshold" (int): Sets the threshold that
            2 * "n_qubits" must be greater than to enable OpenMP
            parallelization for matrix multiplication during execution of
            an experiment. If parallel circuit or shot execution is enabled
            this will only use unallocated CPU cores up to
            max_parallel_threads. Note that setting this too low can reduce
            performance (Default: 14).
    """

    MAX_QUBIT_MEMORY = int(log2(sqrt(local_hardware_info()['memory'] * (1024 ** 3) / 16)))

    DEFAULT_CONFIGURATION = {
        'backend_name': 'unitary_simulator',
        'backend_version': __version__,
        'n_qubits': MAX_QUBIT_MEMORY,
        'url': 'https://github.com/Qiskit/qiskit-aer',
        'simulator': True,
        'local': True,
        'conditional': False,
        'open_pulse': False,
        'memory': False,
        'max_shots': 1,
        'description': 'A Python simulator for computing the unitary'
                       'matrix for experiments in qobj files',
        'coupling_map': None,
        'basis_gates': ['u1', 'u2', 'u3', 'cx', 'cz', 'cu1', 'id', 'x', 'y', 'z',
                        'h', 's', 'sdg', 't', 'tdg', 'ccx', 'swap',
                        'multiplexer', 'snapshot', 'unitary'],
        'gates': [
            {
                'name': 'TODO',
                'parameters': [],
                'qasm_def': 'TODO'
            }
        ],
        # Location where we put external libraries that will be loaded at runtime
        # by the simulator extension
        'library_dir': os.path.dirname(__file__)
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(unitary_controller_execute,
                         BackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION),
                         provider=provider)

    def _validate(self, qobj, backend_options, noise_model):
        """Semantic validations of the qobj which cannot be done via schemas.
        Some of these may later move to backend schemas.
        1. Set shots=1
        2. No measurements or reset
        3. Check number of qubits will fit in local memory.
        """
        name = self.name()
        if noise_model is not None:
            raise AerError("{} does not support noise.".format(name))

        n_qubits = qobj.config.n_qubits
        max_qubits = self.configuration().n_qubits
        if n_qubits > max_qubits:
            raise AerError(
                'Number of qubits ({}) is greater than max ({}) for "{}" with {} GB system memory.'
                .format(n_qubits, max_qubits, name, int(local_hardware_info()['memory'])))
        if qobj.config.shots != 1:
            logger.info('"%s" only supports 1 shot. Setting shots=1.',
                        name)
            qobj.config.shots = 1
        for experiment in qobj.experiments:
            exp_name = experiment.header.name
            if getattr(experiment.config, 'shots', 1) != 1:
                logger.info('"%s" only supports 1 shot. '
                            'Setting shots=1 for circuit "%s".',
                            name, exp_name)
                experiment.config.shots = 1
            for operation in experiment.instructions:
                if operation.name in ['measure', 'reset']:
                    raise AerError(
                        'Unsupported {} instruction {} in circuit {}'
                        .format(name, operation.name, exp_name))
