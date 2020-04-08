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
from math import log2, sqrt
from qiskit.util import local_hardware_info
from qiskit.providers.models import QasmBackendConfiguration

from .aerbackend import AerBackend
from ..aererror import AerError
# pylint: disable=import-error
from .controller_wrappers import unitary_controller_execute
from ..version import __version__

# Logger
logger = logging.getLogger(__name__)


class UnitarySimulator(AerBackend):
    """Ideal quantum circuit unitary simulator.

    **Backend options**

    The following backend options may be used with in the
    ``backend_options`` kwarg for :meth:`UnitarySimulator.run` or
    ``qiskit.execute``.

    * ``"initial_unitary"`` (matrix_like): Sets a custom initial unitary
      matrix for the simulation instead of identity (Default: None).

    * ``"validation_threshold"`` (double): Sets the threshold for checking
      if initial unitary and target unitary are unitary matrices.
      (Default: 1e-8).

    * ``"zero_threshold"`` (double): Sets the threshold for truncating
      small values to zero in the result data (Default: 1e-10).

    * ``"max_parallel_threads"`` (int): Sets the maximum number of CPU
      cores used by OpenMP for parallelization. If set to 0 the
      maximum will be set to the number of CPU cores (Default: 0).

    * ``"max_parallel_experiments"`` (int): Sets the maximum number of
      qobj experiments that may be executed in parallel up to the
      max_parallel_threads value. If set to 1 parallel circuit
      execution will be disabled. If set to 0 the maximum will be
      automatically set to max_parallel_threads (Default: 1).

    * ``"max_memory_mb"`` (int): Sets the maximum size of memory
      to store a state vector. If a state vector needs more, an error
      is thrown. In general, a state vector of n-qubits uses 2^n complex
      values (16 Bytes). If set to 0, the maximum will be automatically
      set to half the system memory size (Default: 0).

    * ``"statevector_parallel_threshold"`` (int): Sets the threshold that
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
        'max_shots': int(1e6),  # Note that this backend will only ever
                                # perform a single shot. This value is just
                                # so that the default shot value for execute
                                # will not raise an error when trying to run
                                # a simulation
        'description': 'A C++ unitary simulator for QASM Qobj files',
        'coupling_map': None,
        'basis_gates': [
            'u1', 'u2', 'u3', 'cx', 'cz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg',
            't', 'tdg', 'swap', 'ccx', 'unitary', 'diagonal', 'cu1', 'cu2',
            'cu3', 'cswap', 'mcx', 'mcy', 'mcz', 'mcu1', 'mcu2', 'mcu3',
            'mcswap', 'multiplexer',
        ],
        'gates': [{
            'name': 'u1',
            'parameters': ['lam'],
            'conditional': True,
            'description': 'Single-qubit gate [[1, 0], [0, exp(1j*lam)]]',
            'qasm_def': 'gate u1(lam) q { U(0,0,lam) q; }'
        }, {
            'name': 'u2',
            'parameters': ['phi', 'lam'],
            'conditional': True,
            'description':
            'Single-qubit gate [[1, -exp(1j*lam)], [exp(1j*phi), exp(1j*(phi+lam))]]/sqrt(2)',
            'qasm_def': 'gate u2(phi,lam) q { U(pi/2,phi,lam) q; }'
        }, {
            'name':
            'u3',
            'parameters': ['theta', 'phi', 'lam'],
            'conditional':
            True,
            'description':
            'Single-qubit gate with three rotation angles',
            'qasm_def':
            'gate u3(theta,phi,lam) q { U(theta,phi,lam) q; }'
        }, {
            'name': 'cx',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit Controlled-NOT gate',
            'qasm_def': 'gate cx c,t { CX c,t; }'
        }, {
            'name': 'cz',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit Controlled-Z gate',
            'qasm_def': 'gate cz a,b { h b; cx a,b; h b; }'
        }, {
            'name': 'id',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit identity gate',
            'qasm_def': 'gate id a { U(0,0,0) a; }'
        }, {
            'name': 'x',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-X gate',
            'qasm_def': 'gate x a { U(pi,0,pi) a; }'
        }, {
            'name': 'y',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-Y gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'z',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Pauli-Z gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'h',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit Hadamard gate',
            'qasm_def': 'TODO'
        }, {
            'name': 's',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit phase gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'sdg',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit adjoint phase gate',
            'qasm_def': 'TODO'
        }, {
            'name': 't',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit T gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'tdg',
            'parameters': [],
            'conditional': True,
            'description': 'Single-qubit adjoint T gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'swap',
            'parameters': [],
            'conditional': True,
            'description': 'Two-qubit SWAP gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'ccx',
            'parameters': [],
            'conditional': True,
            'description': 'Three-qubit Toffoli gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'cswap',
            'parameters': [],
            'conditional': True,
            'description': 'Three-qubit Fredkin (controlled-SWAP) gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'unitary',
            'parameters': ['matrix'],
            'conditional': True,
            'description': 'N-qubit arbitrary unitary gate. '
                           'The parameter is the N-qubit matrix to apply.',
            'qasm_def': 'unitary(matrix) q1, q2,...'
        }, {
            'name': 'diagonal',
            'parameters': ['diag_elements'],
            'conditional': True,
            'description': 'N-qubit diagonal unitary gate. The parameters are the'
                           ' diagonal entries of the N-qubit matrix to apply.',
            'qasm_def': 'TODO'
        }, {
            'name': 'cu1',
            'parameters': ['lam'],
            'conditional': True,
            'description': 'Two-qubit Controlled-u1 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'cu2',
            'parameters': ['phi', 'lam'],
            'conditional': True,
            'description': 'Two-qubit Controlled-u2 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'cu3',
            'parameters': ['theta', 'phi', 'lam'],
            'conditional': True,
            'description': 'Two-qubit Controlled-u3 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcx',
            'parameters': [],
            'conditional': True,
            'description': 'N-qubit multi-controlled-X gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcy',
            'parameters': [],
            'conditional': True,
            'description': 'N-qubit multi-controlled-Y gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcz',
            'parameters': [],
            'conditional': True,
            'description': 'N-qubit multi-controlled-Z gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcu1',
            'parameters': ['lam'],
            'conditional': True,
            'description': 'N-qubit multi-controlled-u1 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcu2',
            'parameters': ['phi', 'lam'],
            'conditional': True,
            'description': 'N-qubit multi-controlled-u2 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcu3',
            'parameters': ['theta', 'phi', 'lam'],
            'conditional': True,
            'description': 'N-qubit multi-controlled-u3 gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'mcswap',
            'parameters': [],
            'conditional': True,
            'description': 'N-qubit multi-controlled-SWAP gate',
            'qasm_def': 'TODO'
        }, {
            'name': 'multiplexer',
            'parameters': ['mat1', 'mat2', '...'],
            'conditional': True,
            'description': 'N-qubit multi-plexer gate. '
                           'The input parameters are the gates for each value.',
            'qasm_def': 'TODO'
        }]
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(unitary_controller_execute,
                         QasmBackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION),
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
