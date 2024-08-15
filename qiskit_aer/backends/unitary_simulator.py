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
Aer Unitary Simulator Backend.
"""
import copy
import logging
from warnings import warn

from qiskit.providers.options import Options

from ..aererror import AerError
from ..version import __version__
from .aerbackend import AerBackend
from .backend_utils import (
    cpp_execute_circuits,
    available_devices,
    MAX_QUBITS_STATEVECTOR,
    LEGACY_METHOD_MAP,
    add_final_save_op,
    map_legacy_method_config,
)
from .backendconfiguration import AerBackendConfiguration

# pylint: disable=import-error, no-name-in-module, abstract-method
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

    * ``max_shot_size`` (int or None): If the number of shots of a noisy
      circuit exceeds this value simulation will be split into multi
      circuits for execution and the results accumulated. If ``None``
      circuits will not be split based on shots. When splitting circuits
      use the ``max_job_size`` option to control how these split circuits
      should be submitted to the executor (Default: None).

    * ``max_shot_size`` (int or None): If the number of shots with
      a noise model exceeds this value, simulation will split the experiments into
      sub experiments. If ``None``  simulator does nothing (Default: None).

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
      experiments that may be executed in parallel up to the
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
        "backend_name": "unitary_simulator",
        "backend_version": __version__,
        "n_qubits": MAX_QUBITS_STATEVECTOR // 2,
        "url": "https://github.com/Qiskit/qiskit-aer",
        "simulator": True,
        "local": True,
        "conditional": False,
        "open_pulse": False,
        "memory": False,
        "max_shots": int(1e6),  # Note that this backend will only ever
        # perform a single shot. This value is just
        # so that the default shot value for execute
        # will not raise an error when trying to run
        # a simulation
        "description": "A C++ unitary circuit simulator",
        "coupling_map": None,
        "basis_gates": sorted(
            [
                "u1",
                "u2",
                "u3",
                "u",
                "p",
                "r",
                "rx",
                "ry",
                "rz",
                "id",
                "x",
                "y",
                "z",
                "h",
                "s",
                "sdg",
                "sx",
                "sxdg",
                "t",
                "tdg",
                "swap",
                "cx",
                "cy",
                "cz",
                "csx",
                "cu",
                "cp",
                "cu1",
                "cu2",
                "cu3",
                "rxx",
                "ryy",
                "rzz",
                "rzx",
                "ccx",
                "ccz",
                "cswap",
                "mcx",
                "mcy",
                "mcz",
                "mcsx",
                "mcu",
                "mcp",
                "mcphase",
                "mcu1",
                "mcu2",
                "mcu3",
                "mcrx",
                "mcry",
                "mcrz",
                "mcr",
                "mcswap",
                "unitary",
                "diagonal",
                "multiplexer",
                "delay",
                "pauli",
            ]
        ),
        "custom_instructions": sorted(["save_unitary", "save_state", "set_unitary", "reset"]),
        "gates": [],
    }

    _SIMULATION_DEVICES = ("CPU", "GPU", "Thrust")

    _AVAILABLE_DEVICES = None

    def __init__(self, configuration=None, properties=None, provider=None, **backend_options):
        warn(
            "The `UnitarySimulator` backend will be deprecated in the"
            " future. It has been superseded by the `AerSimulator`"
            " backend. To obtain legacy functionality initialize with"
            ' `AerSimulator(method="unitary")` and append run circuits'
            " with the `save_state` instruction.",
            PendingDeprecationWarning,
        )

        self._controller = aer_controller_execute()

        if UnitarySimulator._AVAILABLE_DEVICES is None:
            UnitarySimulator._AVAILABLE_DEVICES = available_devices(self._controller)

        if configuration is None:
            configuration = AerBackendConfiguration.from_dict(
                UnitarySimulator._DEFAULT_CONFIGURATION
            )
        else:
            configuration.open_pulse = False

        super().__init__(
            configuration, properties=properties, provider=provider, backend_options=backend_options
        )

    @classmethod
    def _default_options(cls):
        return Options(
            # Global options
            shots=1,
            device="CPU",
            precision="double",
            executor=None,
            max_job_size=None,
            max_shot_size=None,
            zero_threshold=1e-10,
            seed_simulator=None,
            validation_threshold=None,
            max_parallel_threads=None,
            max_parallel_experiments=None,
            max_parallel_shots=None,
            max_memory_mb=None,
            fusion_enable=True,
            fusion_verbose=False,
            fusion_max_qubit=5,
            fusion_threshold=14,
            blocking_qubits=None,
            blocking_enable=False,
            # statevector options
            statevector_parallel_threshold=14,
        )

    def set_option(self, key, value):
        if key == "method":
            # Handle deprecation of method option for device option
            warn(
                "The method option of the `UnitarySimulator` has been"
                " deprecated as of qiskit-aer 0.9.0. To run a GPU statevector"
                " simulation use the option `device='GPU'` instead",
                DeprecationWarning,
            )
            if value in LEGACY_METHOD_MAP:
                value, device = LEGACY_METHOD_MAP[value]
                self.set_option("device", device)
            if value != "unitary":
                raise AerError("only the 'unitary' method is supported for the UnitarySimulator")
            return
        super().set_option(key, value)

    def available_methods(self):
        """Return the available simulation methods."""
        warn(
            "The `available_methods` method of the UnitarySimulator"
            " is deprecated as of qiskit-aer 0.9.0 as this simulator only"
            " supports a single method. To check if GPU simulation is available"
            " use the `available_devices` method instead.",
            DeprecationWarning,
        )
        return ("unitary",)

    def available_devices(self):
        """Return the available simulation methods."""
        return copy.copy(self._AVAILABLE_DEVICES)

    def _execute_circuits(self, aer_circuits, noise_model, config):
        """Execute circuits on the backend."""
        config = map_legacy_method_config(config)
        aer_circuits = add_final_save_op(aer_circuits, "unitary")
        return cpp_execute_circuits(self._controller, aer_circuits, noise_model, config)
