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
import copy
import logging
from warnings import warn
from qiskit.util import local_hardware_info
from qiskit.providers.options import Options
from qiskit.providers.models import QasmBackendConfiguration

from ..aererror import AerError
from ..version import __version__
from .aerbackend import AerBackend
from .backend_utils import (cpp_execute, available_devices,
                            MAX_QUBITS_STATEVECTOR,
                            LEGACY_METHOD_MAP,
                            add_final_save_instruction,
                            map_legacy_method_options)
# pylint: disable=import-error, no-name-in-module
from .controller_wrappers import aer_controller_execute

# Logger
logger = logging.getLogger(__name__)


class UnitarySimulator(AerBackend):
    """Ideal quantum circuit unitary simulator.

    **Configurable Options**

    The `UnitarySimulator` supports CPU and GPU simulation methods and
    additional configurable options. These may be set using the appropriate kwargs
    during initialization. They can also be set of updated using the
    :meth:`set_options` method.

    Run-time options may also be specified as kwargs using the :meth:`run` method.
    These will not be stored in the backend and will only apply to that execution.
    They will also override any previously set options.

    For example, to configure a a single-precision simulator

    .. code-block:: python

        backend = UnitarySimulator(precision='single')

    **Backend Options**

    The following configurable backend options are supported

    * ``device`` (str): Set the simulation device (Default: ``"CPU"``).
      Use :meth:`available_devices` to return a list of devices supported
      on the current system.

    * ``method`` (str): [DEPRECATED] Set the simulation method supported
      methods are ``"unitary"`` for CPU simulation, and
      ``"unitary_gpu"`` for GPU simulation. This option has been
      deprecated, use the ``device`` option to set "CPU" or "GPU"
      simulation instead.

    * ``precision`` (str): Set the floating point precision for
      certain simulation methods to either ``"single"`` or ``"double"``
      precision (default: ``"double"``).

    * ``executor`` (futures.Executor): Set a custom executor for
      asynchronous running of simulation jobs (Default: None).

    * ``max_job_size`` (int or None): If the number of run circuits
      exceeds this value simulation will be run as a set of of sub-jobs
      on the executor. If ``None`` simulation of all circuits are submitted
      to the executor as a single job (Default: None).

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
      set to the system memory size (Default: 0).

    * ``"statevector_parallel_threshold"`` (int): Sets the threshold that
      2 * "n_qubits" must be greater than to enable OpenMP
      parallelization for matrix multiplication during execution of
      an experiment. If parallel circuit or shot execution is enabled
      this will only use unallocated CPU cores up to
      max_parallel_threads. Note that setting this too low can reduce
      performance (Default: 14).

    These backend options apply in circuit optimization passes:

    * ``fusion_enable`` (bool): Enable fusion optimization in circuit
      optimization passes [Default: True]
    * ``fusion_verbose`` (bool): Output gates generated in fusion optimization
      into metadata [Default: False]
    * ``fusion_max_qubit`` (int): Maximum number of qubits for a operation generated
      in a fusion optimization [Default: 5]
    * ``fusion_threshold`` (int): Threshold that number of qubits must be greater
      than or equal to enable fusion optimization [Default: 7]
    """

    _DEFAULT_CONFIGURATION = {
        'backend_name': 'unitary_simulator',
        'backend_version': __version__,
        'n_qubits': MAX_QUBITS_STATEVECTOR // 2,
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
        'description': 'A C++ unitary circuit simulator',
        'coupling_map': None,
        'basis_gates': sorted([
            'u1', 'u2', 'u3', 'u', 'p', 'r', 'rx', 'ry', 'rz', 'id', 'x',
            'y', 'z', 'h', 's', 'sdg', 'sx', 'sxdg', 't', 'tdg', 'swap', 'cx',
            'cy', 'cz', 'csx', 'cu', 'cp', 'cu1', 'cu2', 'cu3', 'rxx', 'ryy',
            'rzz', 'rzx', 'ccx', 'cswap', 'mcx', 'mcy', 'mcz', 'mcsx',
            'mcu', 'mcp', 'mcphase', 'mcu1', 'mcu2', 'mcu3', 'mcrx', 'mcry', 'mcrz',
            'mcr', 'mcswap', 'unitary', 'diagonal', 'multiplexer', 'delay', 'pauli',
        ]),
        'custom_instructions': sorted(['save_unitary', 'save_state', 'set_unitary']),
        'gates': []
    }

    _SIMULATION_DEVICES = ('CPU', 'GPU', 'Thrust')

    _AVAILABLE_DEVICES = None

    def __init__(self,
                 configuration=None,
                 properties=None,
                 provider=None,
                 **backend_options):

        warn('The `UnitarySimulator` backend will be deprecated in the'
             ' future. It has been superseded by the `AerSimulator`'
             ' backend. To obtain legacy functionality initialize with'
             ' `AerSimulator(method="unitary")` and append run circuits'
             ' with the `save_state` instruction.', PendingDeprecationWarning)

        self._controller = aer_controller_execute()

        if UnitarySimulator._AVAILABLE_DEVICES is None:
            UnitarySimulator._AVAILABLE_DEVICES = available_devices(
                self._controller, UnitarySimulator._SIMULATION_DEVICES)

        if configuration is None:
            configuration = QasmBackendConfiguration.from_dict(
                UnitarySimulator._DEFAULT_CONFIGURATION)
        else:
            configuration.open_pulse = False

        super().__init__(configuration,
                         properties=properties,
                         provider=provider,
                         backend_options=backend_options)

    @classmethod
    def _default_options(cls):
        return Options(
            # Global options
            shots=1,
            device="CPU",
            precision="double",
            executor=None,
            max_job_size=None,
            zero_threshold=1e-10,
            seed_simulator=None,
            validation_threshold=None,
            max_parallel_threads=None,
            max_parallel_experiments=None,
            max_parallel_shots=None,
            max_memory_mb=None,
            optimize_ideal_threshold=5,
            optimize_noise_threshold=12,
            fusion_enable=True,
            fusion_verbose=False,
            fusion_max_qubit=5,
            fusion_threshold=14,
            blocking_qubits=None,
            blocking_enable=False,
            # statevector options
            statevector_parallel_threshold=14)

    def set_options(self, **fields):
        if "method" in fields:
            # Handle deprecation of method option for device option
            warn("The method option of the `UnitarySimulator` has been"
                 " deprecated as of qiskit-aer 0.9.0. To run a GPU statevector"
                 " simulation use the option `device='GPU'` instead",
                 DeprecationWarning)
            fields = copy.copy(fields)
            method = fields["method"]
            if method in LEGACY_METHOD_MAP:
                new_method, device = LEGACY_METHOD_MAP[method]
                fields["method"] = new_method
                fields["device"] = device
            if fields["method"] != "unitary":
                raise AerError(
                    "only the 'unitary' method is supported for the UnitarySimulator")
            fields.pop("method")
        super().set_options(**fields)

    def available_methods(self):
        """Return the available simulation methods."""
        warn("The `available_methods` method of the UnitarySimulator"
             " is deprecated as of qiskit-aer 0.9.0 as this simulator only"
             " supports a single method. To check if GPU simulation is available"
             " use the `available_devices` method instead.",
             DeprecationWarning)
        return ("unitary",)

    def available_devices(self):
        """Return the available simulation methods."""
        return copy.copy(self._AVAILABLE_DEVICES)

    def _execute(self, qobj):
        """Execute a qobj on the backend.

        Args:
            qobj (QasmQobj): simulator input.

        Returns:
            dict: return a dictionary of results.
        """
        # Make deepcopy so we don't modify the original qobj
        qobj = copy.deepcopy(qobj)
        qobj = add_final_save_instruction(qobj, "unitary")
        qobj = map_legacy_method_options(qobj)
        return cpp_execute(self._controller, qobj)

    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas.
        Some of these may later move to backend schemas.
        1. Set shots=1
        2. No measurements or reset
        3. Check number of qubits will fit in local memory.
        """
        name = self.name()
        if getattr(qobj.config, 'noise_model', None) is not None:
            raise AerError("{} does not support noise.".format(name))

        n_qubits = qobj.config.n_qubits
        max_qubits = self.configuration().n_qubits
        if n_qubits > max_qubits:
            raise AerError(
                'Number of qubits ({}) is greater than max ({}) for "{}" with {} GB system memory.'
                .format(n_qubits, max_qubits, name,
                        int(local_hardware_info()['memory'])))
        if qobj.config.shots != 1:
            logger.info('"%s" only supports 1 shot. Setting shots=1.', name)
            qobj.config.shots = 1
        for experiment in qobj.experiments:
            exp_name = experiment.header.name
            if getattr(experiment.config, 'shots', 1) != 1:
                logger.info(
                    '"%s" only supports 1 shot. '
                    'Setting shots=1 for circuit "%s".', name, exp_name)
                experiment.config.shots = 1
            for operation in experiment.instructions:
                if operation.name in ['measure', 'reset']:
                    raise AerError(
                        'Unsupported {} instruction {} in circuit {}'.format(
                            name, operation.name, exp_name))
