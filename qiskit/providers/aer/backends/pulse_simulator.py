# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=arguments-differ, missing-return-type-doc

"""
Qiskit Aer pulse simulator backend.
"""

import uuid
import time
import datetime
import logging
from numpy import inf
from qiskit.result import Result
from qiskit.providers.models import BackendConfiguration, PulseDefaults
from .aerbackend import AerBackend
from ..aerjob import AerJob
from ..version import __version__
from ..pulse.controllers.pulse_controller import pulse_controller

logger = logging.getLogger(__name__)


class PulseSimulator(AerBackend):
    r"""Pulse schedule simulator backend.

    The ``PulseSimulator`` simulates continuous time Hamiltonian dynamics of a quantum system,
    with controls specified by pulse :class:`~qiskit.Schedule` objects, and the model of the
    physical system specified by :class:`~qiskit.providers.aer.pulse.PulseSystemModel` objects.
    Results are returned in the same format as when jobs are submitted to actual devices.

    **Example**

    To use the simulator, first :func:`~qiskit.assemble` a :class:`PulseQobj` object
    from a list of pulse :class:`~qiskit.Schedule` objects, using ``backend=PulseSimulator()``.
    Call the simulator with the :class:`PulseQobj` and a
    :class:`~qiskit.providers.aer.pulse.PulseSystemModel` object representing the physical system.

    .. code-block:: python

        backend_sim = qiskit.providers.aer.PulseSimulator()

        # Assemble schedules using PulseSimulator as the backend
        pulse_qobj = assemble(schedules, backend=backend_sim)

        # Run simulation on a PulseSystemModel object
        results = backend_sim.run(pulse_qobj, system_model)

    **Supported PulseQobj parameters**

    * ``qubit_lo_freq``: Local oscillator frequencies for each :class:`DriveChannel`.
      Defaults to either the value given in the
      :class:`~qiskit.providers.aer.pulse.PulseSystemModel`, or is calculated directly
      from the Hamiltonian.
    * ``meas_level``: Type of desired measurement output, in ``[1, 2]``.
      ``1`` gives complex numbers (IQ values), and ``2`` gives discriminated states ``|0>`` and
      ``|1>``. Defaults to ``2``.
    * ``meas_return``: Measurement type, ``'single'`` or ``'avg'``. Defaults to ``'avg'``.
    * ``shots``: Number of shots per experiment. Defaults to ``1024``.


    **Simulation details**

    The simulator uses the ``zvode`` differential equation solver method through ``scipy``.
    Simulation is performed in the rotating frame of the diagonal of the drift Hamiltonian
    contained in the :class:`~qiskit.providers.aer.pulse.PulseSystemModel`. Measurements
    are performed in the `dressed basis` of the drift Hamiltonian.

    **Other options**

    :meth:`PulseSimulator.run` takes an additional ``dict`` argument ``backend_options`` for
    customization. Accepted keys:

    * ``'ode_options'``: A ``dict`` for ``zvode`` solver options. Accepted keys
      are ``'atol'``, ``'rtol'``, ``'nsteps'``, ``'max_step'``, ``'num_cpus'``, ``'norm_tol'``,
      and ``'norm_steps'``.
    """

    DEFAULT_CONFIGURATION = {
        'backend_name': 'pulse_simulator',
        'backend_version': __version__,
        'n_qubits': 20,
        'coupling_map': None,
        'url': 'https://github.com/Qiskit/qiskit-aer',
        'simulator': True,
        'meas_levels': [0, 1, 2],
        'local': True,
        'conditional': True,
        'open_pulse': True,
        'memory': False,
        'max_shots': int(1e6),
        'description': 'A pulse-based Hamiltonian simulator for Pulse Qobj files',
        'gates': [],
        'basis_gates': []
    }

    def __init__(self, configuration=None, provider=None):

        # purpose of defaults is to pass assemble checks
        self._defaults = PulseDefaults(qubit_freq_est=[inf],
                                       meas_freq_est=[inf],
                                       buffer=0,
                                       cmd_def=[],
                                       pulse_library=[])
        super().__init__(self,
                         BackendConfiguration.from_dict(self.DEFAULT_CONFIGURATION),
                         provider=provider)

    def run(self, qobj, system_model, backend_options=None, validate=False):
        """Run a qobj on system_model.

        Args:
            qobj (PulseQobj): Qobj for pulse Schedules to run
            system_model (PulseSystemModel): Physical model to run simulation on
            backend_options (dict): Other options
            validate (bool): Flag for validation checks

        Returns:
            Result: results of simulation
        """
        # Submit job
        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._run_job, qobj, system_model,
                         backend_options, validate)
        aer_job.submit()
        return aer_job

    def _run_job(self, job_id, qobj, system_model, backend_options, validate):
        """Run a qobj job"""
        start = time.time()
        if validate:
            self._validate(qobj, backend_options, noise_model=None)
        # Send problem specification to pulse_controller and get results
        results = pulse_controller(qobj, system_model, backend_options)
        end = time.time()
        return self._format_results(job_id, results, end - start, qobj.qobj_id)

    def _format_results(self, job_id, results, time_taken, qobj_id):
        """Construct Result object from simulator output."""
        # Add result metadata
        output = {}
        output['qobj_id'] = qobj_id
        output['results'] = results
        output['success'] = True
        output["job_id"] = job_id
        output["date"] = datetime.datetime.now().isoformat()
        output["backend_name"] = self.name()
        output["backend_version"] = self.configuration().backend_version
        output["time_taken"] = time_taken
        return Result.from_dict(output)

    def defaults(self):
        """Return defaults.

        Returns:
            PulseDefaults: object for passing assemble.
        """
        return self._defaults
