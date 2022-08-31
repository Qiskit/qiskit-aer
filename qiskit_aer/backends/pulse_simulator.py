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

import copy
import logging
from warnings import warn
from numpy import inf

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import schedule
from qiskit.providers.options import Options
from qiskit.providers.models import BackendConfiguration, PulseDefaults
from qiskit.providers.backend import BackendV2
from qiskit.utils import deprecate_arguments

from ..version import __version__
from ..aererror import AerError
from ..pulse.controllers.pulse_controller import pulse_controller
from ..pulse.system_models.pulse_system_model import PulseSystemModel
from .aerbackend import AerBackend

logger = logging.getLogger(__name__)

DEFAULT_CONFIGURATION = {
    'backend_name': 'pulse_simulator',
    'backend_version': __version__,
    'n_qubits': 20,
    'coupling_map': None,
    'url': 'https://github.com/Qiskit/qiskit-aer',
    'simulator': True,
    'meas_levels': [1, 2],
    'local': True,
    'conditional': True,
    'open_pulse': True,
    'memory': False,
    'max_shots': int(1e6),
    'description': 'A Pulse-based Hamiltonian simulator for Pulse Qobj files',
    'gates': [],
    'basis_gates': [],
    'parametric_pulses': []
}


class PulseSimulator(AerBackend):
    r"""Pulse schedule simulator backend.

    The ``PulseSimulator`` simulates continuous time Hamiltonian dynamics of a quantum system,
    with controls specified by pulse :class:`~qiskit.Schedule` objects, and the model of the
    physical system specified by :class:`~qiskit_aer.pulse.PulseSystemModel` objects.
    Results are returned in the same format as when jobs are submitted to actual devices.

    **Examples**

    The minimal information a ``PulseSimulator`` needs to simulate is a
    :class:`~qiskit_aer.pulse.PulseSystemModel`, which can be supplied either by
    setting the backend option before calling ``run``, e.g.:

    .. code-block:: python

        backend_sim = qiskit_aer.PulseSimulator()

        # Set the pulse system model for the simulator
        backend_sim.set_options(system_model=system_model)

        # Assemble schedules using PulseSimulator as the backend
        pulse_qobj = assemble(schedules, backend=backend_sim)

        # Run simulation
        results = backend_sim.run(pulse_qobj)

    or by supplying the system model at runtime, e.g.:

    .. code-block:: python

        backend_sim = qiskit_aer.PulseSimulator()

        # Assemble schedules using PulseSimulator as the backend
        pulse_qobj = assemble(schedules, backend=backend_sim)

        # Run simulation on a PulseSystemModel object
        results = backend_sim.run(pulse_qobj, system_model=system_model)

    Alternatively, an instance of the ``PulseSimulator`` may be further configured to contain more
    information present in a real backend. The simplest way to do this is to instantiate the
    ``PulseSimulator`` from a real backend:

    .. code-block:: python

        armonk_sim = qiskit_aer.PulseSimulator.from_backend(FakeArmonk())
        pulse_qobj = assemble(schedules, backend=armonk_sim)
        armonk_sim.run(pulse_qobj)

    In the above example, the ``PulseSimulator`` copies all configuration and default data from
    ``FakeArmonk()``, and as such has the same affect as ``FakeArmonk()`` when passed as an
    argument to ``assemble``. Furthermore it constructs a
    :class:`~qiskit_aer.pulse.PulseSystemModel` from the model details in the supplied
    backend, which is then used in simulation.

    **Supported PulseQobj parameters**

    * ``qubit_lo_freq``: Local oscillator frequencies for each :class:`DriveChannel`.
      Defaults to either the value given in the
      :class:`~qiskit_aer.pulse.PulseSystemModel`, or is calculated directly
      from the Hamiltonian.
    * ``meas_level``: Type of desired measurement output, in ``[1, 2]``.
      ``1`` gives complex numbers (IQ values), and ``2`` gives discriminated states ``|0>`` and
      ``|1>``. Defaults to ``2``.
    * ``meas_return``: Measurement type, ``'single'`` or ``'avg'``. Defaults to ``'avg'``.
    * ``shots``: Number of shots per experiment. Defaults to ``1024``.
    * ``executor``: Set a custom executor for asynchronous running of simulation
    * ``max_job_size`` (int or None): If the number of run schedules
      exceeds this value simulation will be run as a set of of sub-jobs
      on the executor. If ``None`` simulation of all schedules are submitted
      to the executor as a single job (Default: None).
    * ``max_shot_size`` (int or None): If the number of shots of a noisy
      circuit exceeds this value simulation will be split into multi
      circuits for execution and the results accumulated. If ``None``
      circuits will not be split based on shots. When splitting circuits
      use the ``max_job_size`` option to control how these split circuits
      should be submitted to the executor (Default: None).

      jobs (Default: None).

    **Simulation details**

    The simulator uses the ``zvode`` differential equation solver method through ``scipy``.
    Simulation is performed in the rotating frame of the diagonal of the drift Hamiltonian
    contained in the :class:`~qiskit_aer.pulse.PulseSystemModel`. Measurements
    are performed in the `dressed basis` of the drift Hamiltonian.

    **Other options**

    Additional valid keyword arguments for ``run()``:

    * ``'solver_options'``: A ``dict`` for solver options. Accepted keys
      are ``'atol'``, ``'rtol'``, ``'nsteps'``, ``'max_step'``, ``'num_cpus'``, ``'norm_tol'``,
      and ``'norm_steps'``.
    """
    def __init__(self,
                 configuration=None,
                 properties=None,
                 defaults=None,
                 provider=None,
                 **backend_options):

        if configuration is None:
            configuration = BackendConfiguration.from_dict(
                DEFAULT_CONFIGURATION)
        else:
            configuration = copy.copy(configuration)
            configuration.meas_levels = self._meas_levels(configuration.meas_levels)
            configuration.open_pulse = True
            configuration.parametric_pulses = []

        if defaults is None:
            defaults = PulseDefaults(qubit_freq_est=[inf],
                                     meas_freq_est=[inf],
                                     buffer=0,
                                     cmd_def=[],
                                     pulse_library=[])

        super().__init__(configuration,
                         properties=properties,
                         defaults=defaults,
                         provider=provider,
                         backend_options=backend_options)

        # Set up default system model
        subsystem_list = backend_options.get('subsystem_list', None)
        if backend_options.get('system_model') is None:
            if hasattr(configuration, 'hamiltonian'):
                system_model = PulseSystemModel.from_config(
                    configuration, subsystem_list)
                self._set_system_model(system_model)

    @classmethod
    def _default_options(cls):
        return Options(
            shots=1024,
            meas_level=None,
            meas_return=None,
            meas_map=None,
            qubit_lo_freq=None,
            solver_options=None,
            subsystem_list=None,
            system_model=None,
            seed=None,
            qubit_freq_est=inf,
            q_level_meas=1,
            noise_model=None,
            initial_state=None,
            executor=None,
            memory_slots=1,
            max_job_size=None,
            max_shot_size=None)

    # pylint: disable=arguments-differ, missing-param-doc
    @deprecate_arguments({'qobj': 'schedules'})
    def run(self,
            schedules,
            validate=True,
            **run_options):
        """Run a qobj on the backend.

        Args:
            schedules (Schedule or list): The pulse :class:`~qiskit.pulse.Schedule`
                (or list of ``Schedule`` objects) to be executed.
            validate (bool): validate the Qobj before running (default: True).
            run_options (kwargs): additional run time backend options.

        Returns:
            AerJob: The simulation job.

        Additional Information:
            * kwarg options specified in ``run_options`` will override options
              of the same kwarg specified in the simulator options, the
              ``backend_options`` and the ``Qobj.config``.
        """
        if isinstance(schedules, list):
            new_schedules = []
            for i in schedules:
                if isinstance(i, QuantumCircuit):
                    new_schedules.append(schedule(i, self))
                else:
                    new_schedules.append(i)
            schedules = new_schedules
        elif isinstance(schedules, QuantumCircuit):
            schedules = schedule(schedules, self)
        return super().run(schedules, validate=validate, **run_options)

    @property
    def _system_model(self):
        return getattr(self._options, 'system_model')

    @classmethod
    def from_backend(cls, backend, **options):
        """Initialize simulator from backend."""
        if isinstance(backend, BackendV2):
            raise AerError(
                "PulseSimulator.from_backend does not currently support V2 Backends."
            )
        configuration = copy.copy(backend.configuration())
        defaults = copy.copy(backend.defaults())
        properties = copy.copy(backend.properties())

        backend_name = 'pulse_simulator({})'.format(configuration.backend_name)
        description = 'A Pulse-based simulator configured from the backend: '
        description += configuration.backend_name

        sim = cls(configuration=configuration,
                  properties=properties,
                  defaults=defaults,
                  backend_name=backend_name,
                  description=description,
                  **options)
        return sim

    def _execute(self, qobj):
        """Execute a qobj on the backend.

        Args:
            qobj (PulseQobj): simulator input.

        Returns:
            dict: return a dictionary of results.
        """
        qobj.config.qubit_freq_est = self.defaults().qubit_freq_est
        return pulse_controller(qobj)

    def set_option(self, key, value):
        """Set pulse simulation options and update backend."""
        if key == 'meas_levels':
            self._set_configuration_option(key, self._meas_levels(value))
            return

        # Handle cases that require updating two places
        if key in ['dt', 'u_channel_lo']:
            self._set_configuration_option(key, value)
            if self._system_model is not None:
                setattr(self._system_model, key, value)
            return

        if key == 'hamiltonian':
            # if option is hamiltonian, set in configuration and reconstruct pulse system model
            subsystem_list = getattr(self._options.get, 'subsystem_list', None)
            system_model = PulseSystemModel.from_config(self.configuration(),
                                                        subsystem_list)
            super().set_option('system_model', system_model)
            self._set_configuration_option(key, value)
            return

        # if system model is specified directly
        if key == 'system_model':
            if hasattr(self.configuration(), 'hamiltonian'):
                warn('Specifying both a configuration with a Hamiltonian and a '
                     'system model may result in inconsistencies.')
            # Set config dt and u_channel_lo to system model values
            self._set_system_model(value)
            return

        if key == 'qubit_lo_freq':
            value = [freq * 1e-9 for freq in value]

        # Set all other options from AerBackend
        super().set_option(key, value)

    def _set_system_model(self, system_model):
        """Set system model option"""
        self._set_configuration_option(
            'dt', getattr(system_model, 'dt', []))
        self._set_configuration_option(
            'u_channel_lo', getattr(system_model, 'u_channel_lo', []))
        super().set_option('system_model', system_model)

    def _validate(self, qobj):
        """Validation of qobj.

        Ensures that exactly one Acquire instruction is present in each
        schedule. Checks SystemModel is in qobj config
        """
        if getattr(qobj.config, 'system_model', None) is None:
            raise AerError("PulseSimulator requires a system model to run.")

        for exp in qobj.experiments:
            num_acquires = 0
            for instruction in exp.instructions:
                if instruction.name == 'acquire':
                    num_acquires += 1

                if num_acquires > 1:
                    raise AerError("PulseSimulator does not support multiple Acquire "
                                   "instructions in a single schedule.")

            if num_acquires == 0:
                raise AerError("PulseSimulator requires at least one Acquire "
                               "instruction per schedule.")

    @staticmethod
    def _meas_levels(meas_levels):
        """Function for setting meas_levels in a pulse simulator configuration."""
        if 0 in meas_levels:
            warn('Measurement level 0 not supported in pulse simulator.')
            tmp = copy.copy(meas_levels)
            tmp.remove(0)
            return tmp
        return meas_levels
