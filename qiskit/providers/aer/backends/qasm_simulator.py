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
import os
from math import log2
from qiskit.util import local_hardware_info
from qiskit.providers.models import BackendConfiguration
from .aerbackend import AerBackend
from .qasm_controller_wrapper import qasm_controller_execute
from ..aererror import AerError
from ..version import __version__

logger = logging.getLogger(__name__)


class QasmSimulator(AerBackend):
    """Aer quantum circuit simulator

    Backend options:

        The following backend options may be used with in the
        `backend_options` kwarg diction for `QasmSimulator.run` or
        `qiskit.execute`

        Simulation method option
        ------------------------
        * "method" (str): Set the simulation method. Allowed values are:
            * "statevector": Uses a dense statevector simulation.
            * "stabilizer": uses a Clifford stabilizer state simulator that
            is only valid for Clifford circuits and noise models.
            * "extended_stabilizer": Uses an approximate simulator that
            decomposes circuits into stabilizer state terms, the number of
            which grows with the number of non-Clifford gates.
            * "automatic": automatically run on stabilizer simulator if
            the circuit and noise model supports it. If there is enough
            available memory, uses the statevector method. Otherwise, uses
            the extended_stabilizer method (Default: "automatic").

        General options
        ---------------

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

        * "max_parallel_shots" (int): Sets the maximum number of
            shots that may be executed in parallel during each experiment
            execution, up to the max_parallel_threads value. If set to 1
            parallel shot execution wil be disabled. If set to 0 the
            maximum will be automatically set to max_parallel_threads.
            Note that this cannot be enabled at the same time as parallel
            experiment execution (Default: 1).

        * "max_memory_mb" (int): Sets the maximum size of memory
            to store a state vector. If a state vector needs more, an error
            is thrown. In general, a state vector of n-qubits uses 2^n complex
            values (16 Bytes). If set to 0, the maximum will be automatically
            set to half the system memory size (Default: 0).

        * "optimize_ideal_threshold" (int): Sets the qubit threshold for
            applying circuit optimization passes on ideal circuits.
            Passes include gate fusion and truncation of unused qubits
            (Default: 5).

        * "optimize_noise_threshold" (int): Sets the qubit threshold for
            applying circuit optimization passes on ideal circuits.
            Passes include gate fusion and truncation of unused qubits
            (Default: 12).

        "statevector" method options
        ----------------------------
        * "statevector_parallel_threshold" (int): Sets the threshold that
            "n_qubits" must be greater than to enable OpenMP
            parallelization for matrix multiplication during execution of
            an experiment. If parallel circuit or shot execution is enabled
            this will only use unallocated CPU cores up to
            max_parallel_threads. Note that setting this too low can reduce
            performance (Default: 14).

        * "statevector_sample_measure_opt" (int): Sets the threshold that
            the number of qubits must be greater than to enable a large
            qubit optimized implementation of measurement sampling. Note
            that setting this two low can reduce performance (Default: 10)

        "stabilizer" method options
        ---------------------------
        * "stabilizer_max_snapshot_probabilities" (int): (Default: 32)

        "extended_stabilizer" method options
        ------------------------------------
        * "extended_stabilizer_measure_sampling" (bool): Enable measure
            sampling optimization on supported circuits. This prevents the
            simulator from re-running the measure monte-carlo step for each
            shot. Enabling measure sampling may reduce accuracy of the
            measurement counts if the output distribution is strongly
            peaked. (Default: False)

        * "extended_stabilizer_mixing_time" (int): Set how long the
            monte-carlo method runs before performing measurements. If the
            output distribution is strongly peaked, this can be decreased
            alongside setting extended_stabilizer_disable_measurement_opt
            to True. (Default: 5000)

        * "extended_stabilizer_approximation_error" (double): Set the error
            in the approximation for the extended_stabilizer method. A
            smaller error needs more memory and computational time.
            (Default: 0.05)

        * "extended_stabilizer_norm_estimation_samples" (int): Number of
            samples used to compute the correct normalisation for a
            statevector snapshot. (Default: 100)

        * "extended_stabilizer_parallel_threshold" (int): Set the minimum
            size of the extended stabilizer decomposition before we enable
            OpenMP parallelisation. If parallel circuit or shot execution
            is enabled this will only use unallocated CPU cores up to
            max_parallel_threads. (Default: 100)
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
        'max_shots': 100000,
        'description': 'A C++ simulator with realistic noise for qobj files',
        'coupling_map': None,
        'basis_gates': [
            'u1', 'u2', 'u3', 'cx', 'cz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg',
            't', 'tdg', 'ccx', 'swap', 'multiplexer', 'snapshot', 'unitary', 'reset',
            'initialize', 'kraus'
        ],
        'gates': [{
            'name': 'TODO',
            'parameters': [],
            'qasm_def': 'TODO'
        }],
        # Location where we put external libraries that will be loaded at runtime
        # by the simulator extension
        'library_dir': os.path.dirname(__file__)
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(
            qasm_controller_execute,
            BackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION),
            provider=provider)

    def _validate(self, qobj, backend_options, noise_model):
        """Semantic validations of the qobj which cannot be done via schemas.

        1. Check number of qubits will fit in local memory.
        2. warn if no classical registers or measurements in circuit.
        """
        clifford_instructions = [
            "id", "x", "y", "z", "h", "s", "sdg", "CX", "cx", "cz", "swap",
            "barrier", "reset", "measure"
        ]
        unsupported_ch_instructions = ["u2", "u3"]
        # Check if noise model is Clifford:
        method = "automatic"
        if backend_options and "method" in backend_options:
            method = backend_options["method"]

        clifford_noise = (method != "statevector")

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
                logger.warning(
                    'No classical registers in circuit "%s": '
                    'result data will not contain counts.', name)
            # Check if Clifford circuit or if measure opts missing
            no_measure = True
            ch_supported = False
            ch_supported = method in ["extended_stabilizer", "automatic"]
            clifford = False if method == "statevector" else clifford_noise
            for op in experiment.instructions:
                if not clifford and not no_measure:
                    break  # we don't need to check any more ops
                if clifford and op.name not in clifford_instructions:
                    clifford = False
                if no_measure and op.name == "measure":
                    no_measure = False
                if ch_supported and op.name in unsupported_ch_instructions:
                    ch_supported = False
            # Print warning if clbits but no measure
            if no_measure:
                logger.warning(
                    'No measurements in circuit "%s": '
                    'count data will return all zeros.', name)
            # Check qubits for statevector simulation
            if not clifford and method != "extended_stabilizer":
                n_qubits = experiment.config.n_qubits
                max_qubits = self.configuration().n_qubits
                if n_qubits > max_qubits:
                    system_memory = int(local_hardware_info()['memory'])
                    err_string = ('Number of qubits ({}) is greater than '
                                  'maximum ({}) for "{}" (method=statevector) '
                                  'with {} GB system memory')
                    err_string = err_string.format(n_qubits, max_qubits,
                                                   self.name(), system_memory)
                    if method != "automatic":
                        raise AerError(err_string + '.')
                    else:
                        if n_qubits > 63:
                            raise AerError('{}, and has too many qubits to fall '
                                           'back to the extended_stabilizer '
                                           'method.'.format(err_string))
                        if not ch_supported:
                            raise AerError('{}, and contains instructions '
                                           'not supported by the extended_etabilizer '
                                           'method.'.format(err_string))
                        logger.info(
                            'The QasmSimulator will automatically '
                            'switch to the Extended Stabilizer backend, based on '
                            'the memory requirements.')
