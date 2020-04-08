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

"""
Qiskit Aer qasm simulator backend.
"""

import logging
from math import log2
from qiskit.util import local_hardware_info
from qiskit.providers.models import QasmBackendConfiguration
from .aerbackend import AerBackend
# pylint: disable=import-error
from .controller_wrappers import qasm_controller_execute
from ..version import __version__

logger = logging.getLogger(__name__)


class QasmSimulator(AerBackend):
    """
    Noisy quantum circuit simulator backend.

    The `QasmSimulator` supports multiple simulation methods and
    configurable options for each simulation method. These options are
    specified in a dictionary which may be passed to the simulator using
    the ``backend_options`` kwarg for :meth:`QasmSimulator.run` or
    ``qiskit.execute``.

    The default behavior chooses a simulation method automatically based on
    the input circuit and noise model. A custom method can be specified using the
    ``"method"`` field in ``backend_options`` as illustrated in the following
    example. Available simulation methods and additional backend options are
    listed below.

    **Example**

    .. code-block:: python

        backend = QasmSimulator()
        backend_options = {"method": "statevector"}

        # Circuit execution
        job = execute(circuits, backend, backend_options=backend_options)

        # Qobj execution
        job = backend.run(qobj, backend_options=backend_options)

    **Simulation method**

    Available simulation methods are:

    * ``"statevector"``: A dense statevector simulation that can sample
      measurement outcomes from *ideal* circuits with all measurements at
      end of the circuit. For noisy simulations each shot samples a
      randomly sampled noisy circuit from the noise model.
      ``"statevector_cpu"`` is an alias of ``"statevector"``.

    * ``"statevector_gpu"``: A dense statevector simulation that provides
      the same functionalities with ``"statevector"``. GPU performs the computation
      to calculate probability amplitudes as CPU does. If no GPU is available,
      a runtime error is raised.

    * ``"density_matrix"``: A dense density matrix simulation that may
      sample measurement outcomes from *noisy* circuits with all
      measurements at end of the circuit. It can only simulate half the
      number of qubits as the statevector method.

    * ``"density_matrix_gpu"``: A dense density matrix simulation that provides
      the same functionalities with ``"density_matrix"``. GPU performs the computation
      to calculate probability amplitudes as CPU does. If no GPU is available,
      a runtime error is raised.

    * ``"stabilizer"``: An efficient Clifford stabilizer state simulator
      that can simulate noisy Clifford circuits if all errors in the noise model are also
      Clifford errors.

    * ``"extended_stabilizer"``: An approximate simulated based on a
      ranked-stabilizer decomposition that decomposes circuits into stabilizer
      state terms. The number of terms grows with the number of
      non-Clifford gates.

    * ``"matrix_product_state"``: A tensor-network statevector simulator that
      uses a Matrix Product State (MPS) representation for the state.

    * ``"automatic"``: The default behavior where the method is chosen
      automatically for each circuit based on the circuit instructions,
      number of qubits, and noise model.

    **Backend options**

    The following backend options may be used with in the
    ``backend_options`` kwarg for :meth:`QasmSimulator.run` or
    ``qiskit.execute``:

    * ``"method"`` (str): Set the simulation method. See backend methods
      for additional information (Default: "automatic").

    * ``"precision"`` (str): Set the floating point precision for
      certain simulation methods to either "single" or "double"
      precision (default: "double").

    * ``"zero_threshold"`` (double): Sets the threshold for truncating
      small values to zero in the result data (Default: 1e-10).

    * ``"validation_threshold"`` (double): Sets the threshold for checking
      if initial states are valid (Default: 1e-8).

    * ``"max_parallel_threads"`` (int): Sets the maximum number of CPU
      cores used by OpenMP for parallelization. If set to 0 the
      maximum will be set to the number of CPU cores (Default: 0).

    * ``"max_parallel_experiments"`` (int): Sets the maximum number of
      qobj experiments that may be executed in parallel up to the
      max_parallel_threads value. If set to 1 parallel circuit
      execution will be disabled. If set to 0 the maximum will be
      automatically set to max_parallel_threads (Default: 1).

    * ``"max_parallel_shots"`` (int): Sets the maximum number of
      shots that may be executed in parallel during each experiment
      execution, up to the max_parallel_threads value. If set to 1
      parallel shot execution will be disabled. If set to 0 the
      maximum will be automatically set to max_parallel_threads.
      Note that this cannot be enabled at the same time as parallel
      experiment execution (Default: 0).

    * ``"max_memory_mb"`` (int): Sets the maximum size of memory
      to store a state vector. If a state vector needs more, an error
      is thrown. In general, a state vector of n-qubits uses 2^n complex
      values (16 Bytes). If set to 0, the maximum will be automatically
      set to half the system memory size (Default: 0).

    * ``"optimize_ideal_threshold"`` (int): Sets the qubit threshold for
      applying circuit optimization passes on ideal circuits.
      Passes include gate fusion and truncation of unused qubits
      (Default: 5).

    * ``"optimize_noise_threshold"`` (int): Sets the qubit threshold for
      applying circuit optimization passes on ideal circuits.
      Passes include gate fusion and truncation of unused qubits
      (Default: 12).

    These backend options only apply when using the ``"statevector"``
    simulation method:

    * ``"statevector_parallel_threshold"`` (int): Sets the threshold that
      the number of qubits must be greater than to enable OpenMP
      parallelization for matrix multiplication during execution of
      an experiment. If parallel circuit or shot execution is enabled
      this will only use unallocated CPU cores up to
      max_parallel_threads. Note that setting this too low can reduce
      performance (Default: 14).

    * ``"statevector_sample_measure_opt"`` (int): Sets the threshold that
      the number of qubits must be greater than to enable a large
      qubit optimized implementation of measurement sampling. Note
      that setting this two low can reduce performance (Default: 10)

    These backend options only apply when using the ``"stabilizer"``
    simulation method:

    * ``"stabilizer_max_snapshot_probabilities"`` (int): set the maximum
      qubit number for the
      `~qiskit.providers.aer.extensions.SnapshotProbabilities`
      instruction (Default: 32).

    These backend options only apply when using the ``"extended_stabilizer"``
    simulation method:

    * ``"extended_stabilizer_measure_sampling"`` (bool): Enable measure
      sampling optimization on supported circuits. This prevents the
      simulator from re-running the measure monte-carlo step for each
      shot. Enabling measure sampling may reduce accuracy of the
      measurement counts if the output distribution is strongly
      peaked (Default: False).

    * ``"extended_stabilizer_mixing_time"`` (int): Set how long the
      monte-carlo method runs before performing measurements. If the
      output distribution is strongly peaked, this can be decreased
      alongside setting extended_stabilizer_disable_measurement_opt
      to True (Default: 5000).

    * ``"extended_stabilizer_approximation_error"`` (double): Set the error
      in the approximation for the extended_stabilizer method. A
      smaller error needs more memory and computational time
      (Default: 0.05).

    * ``"extended_stabilizer_norm_estimation_samples"`` (int): Number of
      samples used to compute the correct normalization for a
      statevector snapshot (Default: 100).

    * ``"extended_stabilizer_parallel_threshold"`` (int): Set the minimum
      size of the extended stabilizer decomposition before we enable
      OpenMP parallelization. If parallel circuit or shot execution
      is enabled this will only use unallocated CPU cores up to
      max_parallel_threads (Default: 100).

    These backend options apply in circuit optimization passes:

    * ``"fusion_enable"`` (bool): Enable fusion optimization in circuit
      optimization passes [Default: True]
    * ``"fusion_verbose"`` (bool): Output gates generated in fusion optimization
      into metadata [Default: False]
    * ``"fusion_max_qubit"`` (int): Maximum number of qubits for a operation generated
      in a fusion optimization [Default: 5]
    * ``"fusion_threshold"`` (int): Threshold that number of qubits must be greater
      than or equal to enable fusion optimization [Default: 20]
    """

    MAX_QUBIT_MEMORY = int(
        log2(local_hardware_info()['memory'] * (1024**3) / 16))

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
        'max_shots': int(1e6),
        'description': 'A C++ simulator with realistic noise for QASM Qobj files',
        'coupling_map': None,
        'basis_gates': [
            'u1', 'u2', 'u3', 'cx', 'cz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg',
            't', 'tdg', 'swap', 'ccx', 'unitary', 'diagonal', 'initialize',
            'cu1', 'cu2', 'cu3', 'cswap', 'mcx', 'mcy', 'mcz',
            'mcu1', 'mcu2', 'mcu3', 'mcswap', 'multiplexer', 'kraus', 'roerror'
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
            'description': 'N-qubit unitary gate. '
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
            'name': 'initialize',
            'parameters': ['vector'],
            'conditional': False,
            'description': 'N-qubit state initialize. '
                           'Resets qubits then sets statevector to the parameter vector.',
            'qasm_def': 'initialize(vector) q1, q2,...'
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
        }, {
            'name': 'kraus',
            'parameters': ['mat1', 'mat2', '...'],
            'conditional': True,
            'description': 'N-qubit Kraus error instruction. '
                           'The input parameters are the Kraus matrices.',
            'qasm_def': 'TODO'
        }, {
            'name': 'roerror',
            'parameters': ['matrix'],
            'conditional': False,
            'description': 'N-bit classical readout error instruction. '
                           'The input parameter is the readout error probability matrix.',
            'qasm_def': 'TODO'
        }]
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(
            qasm_controller_execute,
            QasmBackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION),
            provider=provider)

    def _validate(self, qobj, backend_options, noise_model):
        """Semantic validations of the qobj which cannot be done via schemas.

        Warn if no measurements in circuit with classical registers.
        """
        for experiment in qobj.experiments:
            # If circuit contains classical registers but not
            # measurements raise a warning
            if experiment.config.memory_slots > 0:
                # Check if measure opts missing
                no_measure = True
                for op in experiment.instructions:
                    if not no_measure:
                        break  # we don't need to check any more ops
                    if no_measure and op.name == "measure":
                        no_measure = False
                # Print warning if clbits but no measure
                if no_measure:
                    logger.warning(
                        'No measurements in circuit "%s": '
                        'count data will return all zeros.',
                        experiment.header.name)
