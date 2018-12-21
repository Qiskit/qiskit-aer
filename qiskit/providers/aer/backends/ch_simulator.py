# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Qiskit Aer qasm simulator backend.
"""

from math import log2
from qiskit._util import local_hardware_info
from qiskit.providers.models import BackendConfiguration
from .aerbackend import AerBackend
from ch_controller_wrapper import ch_controller_execute
from ..version import __version__


class CHSimulator(AerBackend):
    """Aer quantum circuit simulator

    Backend options:

        The following backend options may be used with in the
        `backend_options` kwarg diction for `QasmSimulator.run` or
        `qiskit.execute`

        * "srank_approximation_error" (double): Set the maximum allowed error in
            the approximate decomposition of the circuit state. The runtime of
            the simulator scales as err^{-2}. (Default: 0.05)

        * "srank_parallel_threshold" (int): Threshold number of terms in the
            stabilizer rank decoposition before we parallelise. (Default: 100)

        * "srank_mixing_time" (int): Number of steps we run of the metropolis method
            before sampling output strings. (Default: 7000)

        * "srank_norm_estimation_samples" (int): Number of samples used by the
            Norm Estimation algorithm. This is used to normalise the
            state vector. (Default: 100)

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

    DEFAULT_CONFIGURATION = {
        'backend_name': 'qasm_simulator',
        'backend_version': __version__,
        'n_qubits': 63,
        'url': 'TODO',
        'simulator': True,
        'local': True,
        'conditional': True,
        'open_pulse': False,
        'memory': True,
        'max_shots': 100000,
        'description': 'A C++ simulator that decomposes circuits into stabilizer'
                       'states.',
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
        super().__init__(ch_controller_execute,
                         BackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION),
                         provider=provider)

    def _validate(self, qobj):
        # TODO
        return
