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

import copy
import logging
from qiskit.providers.models import QasmBackendConfiguration

from ..version import __version__
from .aerbackend import AerBackend
from .backend_utils import (cpp_execute, available_methods,
                            MAX_QUBITS_STATEVECTOR)
# pylint: disable=import-error, no-name-in-module
from .controller_wrappers import qasm_controller_execute

logger = logging.getLogger(__name__)


class QasmSimulator(AerBackend):
    """
    Noisy quantum circuit simulator backend.

    **Configurable Options**

    The `QasmSimulator` supports multiple simulation methods and
    configurable options for each simulation method. These may be set using the
    appropriate kwargs during initialization. They can also be set of updated
    using the :meth:`set_options` method.

    Run-time options may also be specified as kwargs using the :meth:`run` method.
    These will not be stored in the backend and will only apply to that execution.
    They will also override any previously set options.

    For example, to configure a density matrix simulator with a custom noise
    model to use for every execution

    .. code-block:: python

        noise_model = NoiseModel.from_backend(backend)
        backend = QasmSimulator(method='density_matrix',
                                noise_model=noise_model)

    **Simulating an IBMQ Backend**

    The simulator can be automatically configured to mimic an IBMQ backend using
    the :meth:`from_backend` method. This will configure the simulator to use the
    basic device :class:`NoiseModel` for that backend, and the same basis gates
    and coupling map.

    .. code-block:: python

        backend = QasmSimulator.from_backend(backend)

    **Simulation Method Option**

    The simulation method is set using the ``method`` kwarg.
    Supported simulation methods are:

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

    **Additional Backend Options**

    The following simulator specific backend options are supported

    * ``method`` (str): Set the simulation method (Default: ``"automatic"``).

    * ``precision`` (str): Set the floating point precision for
      certain simulation methods to either ``"single"`` or ``"double"``
      precision (default: ``"double"``).

    * ``zero_threshold`` (double): Sets the threshold for truncating
      small values to zero in the result data (Default: 1e-10).

    * ``validation_threshold`` (double): Sets the threshold for checking
      if initial states are valid (Default: 1e-8).

    * ``max_parallel_threads`` (int): Sets the maximum number of CPU
      cores used by OpenMP for parallelization. If set to 0 the
      maximum will be set to the number of CPU cores (Default: 0).

    * ``max_parallel_experiments`` (int): Sets the maximum number of
      qobj experiments that may be executed in parallel up to the
      max_parallel_threads value. If set to 1 parallel circuit
      execution will be disabled. If set to 0 the maximum will be
      automatically set to max_parallel_threads (Default: 1).

    * ``max_parallel_shots`` (int): Sets the maximum number of
      shots that may be executed in parallel during each experiment
      execution, up to the max_parallel_threads value. If set to 1
      parallel shot execution will be disabled. If set to 0 the
      maximum will be automatically set to max_parallel_threads.
      Note that this cannot be enabled at the same time as parallel
      experiment execution (Default: 0).

    * ``max_memory_mb`` (int): Sets the maximum size of memory
      to store a state vector. If a state vector needs more, an error
      is thrown. In general, a state vector of n-qubits uses 2^n complex
      values (16 Bytes). If set to 0, the maximum will be automatically
      set to half the system memory size (Default: 0).

    * ``optimize_ideal_threshold`` (int): Sets the qubit threshold for
      applying circuit optimization passes on ideal circuits.
      Passes include gate fusion and truncation of unused qubits
      (Default: 5).

    * ``optimize_noise_threshold`` (int): Sets the qubit threshold for
      applying circuit optimization passes on ideal circuits.
      Passes include gate fusion and truncation of unused qubits
      (Default: 12).

    These backend options only apply when using the ``"statevector"``
    simulation method:

    * ``statevector_parallel_threshold`` (int): Sets the threshold that
      the number of qubits must be greater than to enable OpenMP
      parallelization for matrix multiplication during execution of
      an experiment. If parallel circuit or shot execution is enabled
      this will only use unallocated CPU cores up to
      max_parallel_threads. Note that setting this too low can reduce
      performance (Default: 14).

    * ``statevector_sample_measure_opt`` (int): Sets the threshold that
      the number of qubits must be greater than to enable a large
      qubit optimized implementation of measurement sampling. Note
      that setting this two low can reduce performance (Default: 10)

    These backend options only apply when using the ``"stabilizer"``
    simulation method:

    * ``stabilizer_max_snapshot_probabilities`` (int): set the maximum
      qubit number for the
      `~qiskit.providers.aer.extensions.SnapshotProbabilities`
      instruction (Default: 32).

    These backend options only apply when using the ``"extended_stabilizer"``
    simulation method:

    * ``extended_stabilizer_measure_sampling`` (bool): Enable measure
      sampling optimization on supported circuits. This prevents the
      simulator from re-running the measure monte-carlo step for each
      shot. Enabling measure sampling may reduce accuracy of the
      measurement counts if the output distribution is strongly
      peaked (Default: False).

    * ``extended_stabilizer_mixing_time`` (int): Set how long the
      monte-carlo method runs before performing measurements. If the
      output distribution is strongly peaked, this can be decreased
      alongside setting extended_stabilizer_disable_measurement_opt
      to True (Default: 5000).

    * ``"extended_stabilizer_approximation_error"`` (double): Set the error
      in the approximation for the extended_stabilizer method. A
      smaller error needs more memory and computational time
      (Default: 0.05).

    * ``extended_stabilizer_norm_estimation_samples`` (int): Number of
      samples used to compute the correct normalization for a
      statevector snapshot (Default: 100).

    * ``extended_stabilizer_parallel_threshold`` (int): Set the minimum
      size of the extended stabilizer decomposition before we enable
      OpenMP parallelization. If parallel circuit or shot execution
      is enabled this will only use unallocated CPU cores up to
      max_parallel_threads (Default: 100).

    These backend options only apply when using the ``"matrix_product_state"``
    simulation method:

    * ``matrix_product_state_max_bond_dimension`` (int): Sets a limit
      on the number of Schmidt coefficients retained at the end of
      the svd algorithm. Coefficients beyond this limit will be discarded.
      (Default: None, i.e., no limit on the bond dimension).

    * ``matrix_product_state_truncation_threshold`` (double):
      Discard the smallest coefficients for which the sum of
      their squares is smaller than this threshold.
      (Default: 1e-16).

    * ``mps_sample_measure_algorithm`` (str):
      Choose which algorithm to use for ``"sample_measure"``. ``"mps_probabilities"``
      means all state probabilities are computed and measurements are based on them.
      It is more efficient for a large number of shots, small number of qubits and low
      entanglement. ``"mps_apply_measure"`` creates a copy of the mps structure and
      makes a measurement on it. It is more effients for a small number of shots, high
      number of qubits, and low entanglement. If the user does not specify the algorithm,
      a heuristic algorithm is used to select between the two algorithms.
      (Default: "mps_heuristic").

    These backend options apply in circuit optimization passes:

    * ``fusion_enable`` (bool): Enable fusion optimization in circuit
      optimization passes [Default: True]
    * ``fusion_verbose`` (bool): Output gates generated in fusion optimization
      into metadata [Default: False]
    * ``fusion_max_qubit`` (int): Maximum number of qubits for a operation generated
      in a fusion optimization [Default: 5]
    * ``fusion_threshold`` (int): Threshold that number of qubits must be greater
      than or equal to enable fusion optimization [Default: 14]
    """

    _DEFAULT_CONFIGURATION = {
        'backend_name': 'qasm_simulator',
        'backend_version': __version__,
        'n_qubits': MAX_QUBITS_STATEVECTOR,
        'url': 'https://github.com/Qiskit/qiskit-aer',
        'simulator': True,
        'local': True,
        'conditional': True,
        'open_pulse': False,
        'memory': True,
        'max_shots': int(1e6),
        'description': 'A C++ QasmQobj simulator with noise',
        'coupling_map': None,
        'basis_gates': sorted([
            'u1', 'u2', 'u3', 'u', 'p', 'r', 'rx', 'ry', 'rz', 'id', 'x',
            'y', 'z', 'h', 's', 'sdg', 'sx', 't', 'tdg', 'swap', 'cx',
            'cy', 'cz', 'csx', 'cp', 'cu1', 'cu2', 'cu3', 'rxx', 'ryy',
            'rzz', 'rzx', 'ccx', 'cswap', 'mcx', 'mcy', 'mcz', 'mcsx',
            'mcp', 'mcu1', 'mcu2', 'mcu3', 'mcrx', 'mcry', 'mcrz',
            'mcr', 'mcswap', 'unitary', 'diagonal', 'multiplexer',
            'initialize', 'delay',
            # Custom instructions
            'kraus', 'roerror', 'snapshot'
        ]),
        'custom_instructions': sorted(['roerror', 'kraus', 'snapshot']),
        'gates': []
    }

    _AVAILABLE_METHODS = None

    def __init__(self,
                 configuration=None,
                 properties=None,
                 provider=None,
                 **backend_options):

        self._controller = qasm_controller_execute()

        # Update available methods for class
        if QasmSimulator._AVAILABLE_METHODS is None:
            QasmSimulator._AVAILABLE_METHODS = available_methods(
                self._controller, [
                    'automatic', 'statevector', 'statevector_gpu',
                    'statevector_thrust', 'density_matrix',
                    'density_matrix_gpu', 'density_matrix_thrust',
                    'stabilizer', 'matrix_product_state', 'extended_stabilizer'
                ])

        if configuration is None:
            configuration = self._method_configuration()
        elif not hasattr(configuration, 'custom_instructions'):
            configuration.custom_instructions = []

        super().__init__(configuration,
                         properties=properties,
                         available_methods=QasmSimulator._AVAILABLE_METHODS,
                         provider=provider,
                         backend_options=backend_options)

    @classmethod
    def from_backend(cls, backend, **options):
        """Initialize simulator from backend."""
        # pylint: disable=import-outside-toplevel
        # Avoid cyclic import
        from ..noise.noise_model import NoiseModel

        # Get configuration and properties from backend
        configuration = copy.copy(backend.configuration())
        properties = copy.copy(backend.properties())

        # Customize configuration name
        name = configuration.backend_name
        configuration.backend_name = 'qasm_simulator({})'.format(name)

        # Basis gates and Custom instructions
        basis_gates = set(configuration.basis_gates)
        custom_instr = cls._DEFAULT_CONFIGURATION['custom_instructions']
        configuration.custom_instructions = sorted(custom_instr)
        configuration.basis_gates = sorted(basis_gates.union(custom_instr))

        # Use automatic noise model if none is provided
        if 'noise_model' not in options:
            noise_model = NoiseModel.from_backend(backend)
            if not noise_model.is_ideal():
                options['noise_model'] = noise_model

        # Initialize simulator
        sim = cls(configuration=configuration,
                  properties=properties,
                  **options)
        return sim

    def _execute(self, qobj):
        """Execute a qobj on the backend.

        Args:
            qobj (QasmQobj): simulator input.

        Returns:
            dict: return a dictionary of results.
        """
        return cpp_execute(self._controller, qobj)

    def _set_option(self, key, value):
        """Set the simulation method and update configuration.

        Args:
            key (str): key to update
            value (any): value to update.

        Raises:
            AerError: if key is 'method' and val isn't in available methods.
        """
        # If key is noise_model we also change the simulator config
        # to use the noise_model basis gates by default.
        if key == 'noise_model' and value is not None:
            basis_gates = set(self._configuration.basis_gates)  # Method basis gates
            intersection = basis_gates.intersection(value.basis_gates)
            self._check_basis_gates(basis_gates, value.basis_gates, intersection)
            self._set_option('basis_gates', intersection)

        # If key is method we update our configurations
        if key == 'method':
            method_config = self._method_configuration(value)
            self._set_configuration_option('description', method_config.description)
            self._set_configuration_option('backend_name', method_config.backend_name)
            self._set_configuration_option('n_qubits', method_config.n_qubits)
            self._set_configuration_option('custom_instructions',
                                           method_config.custom_instructions)
            # Take intersection of method basis gates with configuration
            # basis gates and noise model basis gates
            basis_gates = set(self._configuration.basis_gates)
            basis_gates = basis_gates.intersection(method_config.basis_gates)
            if 'noise_model' in self.options:
                noise_gates = self.options['noise_model'].basis_gates
                intersection = basis_gates.intersection(noise_gates)
                self._check_basis_gates(basis_gates, noise_gates, intersection)
                basis_gates = intersection
            self._set_option('basis_gates', basis_gates)

        # When setting basis gates always append custom simulator instructions for
        # the current method
        if key == 'basis_gates':
            value = sorted(set(value).union(self.configuration().custom_instructions))

        # Set all other options from AerBackend
        super()._set_option(key, value)

    @staticmethod
    def _check_basis_gates(method_gates, noise_gates, intersection=None):
        """Check if intersection of method basis gates and noise basis gates is empty"""
        if intersection is None:
            intersection = set(method_gates).intersection(noise_gates)
        if not intersection:
            logger.warning(
                "The intersection of NoiseModel basis gates (%s) and "
                "backend basis gates (%s) is empty",
                sorted(noise_gates), sorted(method_gates))

    def _validate(self, qobj):
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

    @staticmethod
    def _method_configuration(method=None):
        """Return QasmBackendConfiguration."""
        # Default configuration
        config = QasmBackendConfiguration.from_dict(
            QasmSimulator._DEFAULT_CONFIGURATION)

        # Statevector methods
        if method in ['statevector', 'statevector_gpu', 'statevector_thrust']:
            config.description = 'A C++ QasmQobj statevector simulator with noise'

        # Density Matrix methods
        elif method in [
                'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust'
        ]:
            config.n_qubits = config.n_qubits // 2
            config.description = 'A C++ QasmQobj density matrix simulator with noise'
            config.custom_instructions = sorted(['roerror', 'snapshot', 'kraus', 'superop'])
            config.basis_gates = sorted([
                'u1', 'u2', 'u3', 'u', 'p', 'r', 'rx', 'ry', 'rz', 'id', 'x',
                'y', 'z', 'h', 's', 'sdg', 'sx', 't', 'tdg', 'swap', 'cx',
                'cy', 'cz', 'cp', 'cu1', 'rxx', 'ryy', 'rzz', 'rzx', 'ccx',
                'unitary', 'diagonal', 'delay',
            ] + config.custom_instructions)

        # Matrix product state method
        elif method == 'matrix_product_state':
            config.description = 'A C++ QasmQobj matrix product state simulator with noise'
            config.custom_instructions = sorted(['roerror', 'snapshot', 'kraus'])
            config.basis_gates = sorted([
                'u1', 'u2', 'u3', 'u', 'p', 'cp', 'cx', 'cz', 'id', 'x', 'y', 'z', 'h', 's',
                'sdg', 'sx', 't', 'tdg', 'swap', 'ccx', 'unitary', 'delay'
            ] + config.custom_instructions)

        # Stabilizer method
        elif method == 'stabilizer':
            config.n_qubits = 5000  # TODO: estimate from memory
            config.description = 'A C++ QasmQobj Clifford stabilizer simulator with noise'
            config.custom_instructions = sorted(['roerror', 'snapshot'])
            config.basis_gates = sorted([
                'id', 'x', 'y', 'z', 'h', 's', 'sdg', 'sx', 'cx', 'cy', 'cz',
                'swap', 'delay',
            ] + config.custom_instructions)

        # Extended stabilizer method
        elif method == 'extended_stabilizer':
            config.n_qubits = 63  # TODO: estimate from memory
            config.description = 'A C++ QasmQobj ranked stabilizer simulator with noise'
            config.custom_instructions = sorted(['roerror', 'snapshot'])
            config.basis_gates = sorted([
                'cx', 'cz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 'sx', 'swap',
                'u0', 'u1', 'p', 'ccx', 'ccz', 'delay'
            ] + config.custom_instructions)

        return config
