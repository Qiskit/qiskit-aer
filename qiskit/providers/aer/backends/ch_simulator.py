# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Qiskit Aer qasm simulator backend.
"""

import datetime
import logging
import json

from qiskit._util import local_hardware_info
from qiskit.providers.models import BackendConfiguration

from qiskit.result import Result

from .aerbackend import AerBackend
from ..aererror import AerError
from ch_controller_wrapper import ch_controller_execute, ch_validate_memory
from ..version import __version__

logger = logging.getLogger(__name__)

class CHSimulator(AerBackend):
    """Aer quantum circuit simulator using the CH Representation

    Backend options:

        The following backend options may be used with in the
        `backend_options` kwarg diction for `QasmSimulator.run` or
        `qiskit.execute`

        * "srank_approximation_error" (double): Set the maximum allowed error in
            the approximate decomposition of the circuit state. The runtime of
            the simulator scales as err^{-2}. (Default: 0.05)

        * "srank_parallel_threshold" (int): Threshold number of terms in the
            stabilizer rank decoposition before we parallelise. (Default: 100)

        * "srank_mixing_time" (int): Number of steps we run of the metropolis
            method before sampling output strings. (Default: 7000)

        * "srank_norm_estimation_samples" (int): Number of samples used by the
            Norm Estimation algorithm. This is used to normalise the
            state vector. (Default: 100)

        * "disable_measurement_opt" (bool): Flag that controls if we
            use an 'optimised' measurement method that 'mixes' the monte carlo
            estimator once, before sampling `shots` times. This significantly
            reduces the computational time needed to sample from the output
            distribution, but performs poorly on strongly peaked probability
            distributions as it can become stuck in local maxima.
            (Default: False)

        * "probabilities_snapshot_samples" (int): Number of output strings we
            sample to estimate output probability. (Default: 3000)

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
    """

    MAX_MEMORY = local_hardware_info()['memory']

    DEFAULT_CONFIGURATION = {
        'backend_name': 'ch_simulator',
        'backend_version': __version__,
        'n_qubits': 63,
        'url': 'https://github.com/padraic-padraic/qiskit-aer',
        'simulator': True,
        'local': True,
        'conditional': True,
        'open_pulse': False,
        'memory': True,
        'max_shots': 100000,
        'description': 'A C++ simulator that decomposes circuits into '
                        'stabilizer states.',
        'basis_gates': ['u1', 'cx', 'cz', 'id', 'x', 'y', 'z',
                        'h', 's', 'sdg', 't', 'tdg', 'ccx', 'ccz', 'swap',
                        'snapshot'],
        'gates': [
            {
                'name': 'TODO',
                'parameters': [],
                'qasm_def': 'TODO'
            }
        ]
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(ch_controller_execute,
                         BackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION),
                         provider=provider)

    def _validate(self, qobj, backend_options, noise_model):
        clifford_instructions = ["id", "x", "y", "z", "h", "s", "sdg",
                                 "CX", "cx", "cz", "swap",
                                 "barrier", "reset", "measure"]
        # Check if noise model is Clifford:
        name = self.name()
        if noise_model:
            for error in noise_model.as_dict()['errors']:
                if error['type'] == 'qerror':
                    for circ in error["instructions"]:
                        for instr in circ:
                            if instr not in clifford_instructions:
                                raise AerError('{} does not support '
                                               'non-Clifford '
                                               ' noise'.format(name))
                                break
        # Check to see if experiments are clifford
        all_clifford = True
        for experiment in qobj.experiments:
            name = experiment.header.name
            # Check if Clifford circuit or if measure opts missing
            no_measure = True
            for op in experiment.instructions:
                if not all_clifford and not no_measure:
                    break  # we don't need to check any more ops
                if all_clifford and op.name not in clifford_instructions:
                    all_clifford = False
                if no_measure and op.name == "measure":
                    no_measure = False
            # Print warning if clbits but no measure
            if no_measure:
                logger.warning('No measurements in circuit "%s": '
                               'count data will return all zeros.', name)
            # Check qubits for statevector simulation
            if experiment.config.n_qubits > self.configuration().n_qubits:
                raise AerError('Number of qubits({}) is greater than the'
                               'maximum of 63 currently supported by the'
                               'CH backend.')
        if not all_clifford:
            qobj_str = self._format_qobj_str(qobj, backend_options,
                                             noise_model)
            sufficient_memory = ch_validate_memory(qobj_str, self.MAX_MEMORY)
            if not sufficient_memory:
                accuracy = self.configuration.get(
                            'srank_approximation_error',
                            0.05)
                raise AerError('The number of terms required to simulate '
                               ' these circuits with the desired precision'
                               ' {} would require more memory than is '
                               'currently available'.format(accuracy))
