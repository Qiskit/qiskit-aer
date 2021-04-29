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
from warnings import warn
from qiskit.providers.options import Options
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
    Supported simulation methods are

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
      set to the system memory size (Default: 0).

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

    * ``extended_stabilizer_sampling_methid`` (string): Choose how to simulate
      measurements on qubits. The performance of the simulator depends
      significantly on this choice. In the following, let n be the number of
      qubits in the circuit, m the number of qubits measured, and S be the
      number of shots. (Default: resampled_metropolis)

      * ``"metropolis"``: Use a Monte-Carlo method to sample many output
        strings from the simulator at once. To be accurate, this method
        requires that all the possible output strings have a non-zero
        probability. It will give inaccurate results on cases where
        the circuit has many zero-probability outcomes.
        This method has an overall runtime that scales as n^{2} + (S-1)n.

      * ``"resampled_metropolis"``: A variant of the metropolis method,
        where the Monte-Carlo method is reinitialised for every shot. This
        gives better results for circuits where some outcomes have zero
        probability, but will still fail if the output distribution
        is sparse. The overall runtime scales as Sn^{2}.

      * ``"norm_estimation"``: An alternative sampling method using
        random state inner products to estimate outcome probabilites. This
        method requires twice as much memory, and significantly longer
        runtimes, but gives accurate results on circuits with sparse
        output distributions. The overall runtime scales as Sn^{3}m^{3}.

    * ``extended_stabilizer_metropolis_mixing_time`` (int): Set how long the
      monte-carlo method runs before performing measurements. If the
      output distribution is strongly peaked, this can be decreased
      alongside setting extended_stabilizer_disable_measurement_opt
      to True (Default: 5000).

    * ``"extended_stabilizer_approximation_error"`` (double): Set the error
      in the approximation for the extended_stabilizer method. A
      smaller error needs more memory and computational time
      (Default: 0.05).

    * ``extended_stabilizer_norm_estimation_samples`` (int): The default number
      of samples for the norm estimation sampler. The method will use the
      default, or 4m^{2} samples where m is the number of qubits to be
      measured, whichever is larger (Default: 100).

    * ``extended_stabilizer_norm_estimation_repetitions`` (int): The number
      of times to repeat the norm estimation. The median of these reptitions
      is used to estimate and sample output strings (Default: 3).

    * ``extended_stabilizer_parallel_threshold`` (int): Set the minimum
      size of the extended stabilizer decomposition before we enable
      OpenMP parallelization. If parallel circuit or shot execution
      is enabled this will only use unallocated CPU cores up to
      max_parallel_threads (Default: 100).

    * ``extended_stabilizer_probabilities_snapshot_samples`` (int): If using
      the metropolis or resampled_metropolis sampling method, set the number of
      samples used to estimate probabilities in a probabilities snapshot
      (Default: 3000).

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

    _DEFAULT_BASIS_GATES = sorted([
        'u1', 'u2', 'u3', 'u', 'p', 'r', 'rx', 'ry', 'rz', 'id', 'x',
        'y', 'z', 'h', 's', 'sdg', 'sx', 't', 'tdg', 'swap', 'cx',
        'cy', 'cz', 'csx', 'cp', 'cu1', 'cu2', 'cu3', 'rxx', 'ryy',
        'rzz', 'rzx', 'ccx', 'cswap', 'mcx', 'mcy', 'mcz', 'mcsx',
        'mcphase', 'mcu1', 'mcu2', 'mcu3', 'mcrx', 'mcry', 'mcrz',
        'mcr', 'mcswap', 'unitary', 'diagonal', 'multiplexer',
        'initialize', 'delay', 'pauli', 'mcx_gray'
    ])

    _DEFAULT_CUSTOM_INSTR = sorted([
        'roerror', 'kraus', 'snapshot', 'save_expval', 'save_expval_var',
        'save_probabilities', 'save_probabilities_dict',
        'save_amplitudes', 'save_amplitudes_sq', 'save_state',
        'save_density_matrix', 'save_statevector', 'save_statevector_dict',
        'save_stabilizer', 'set_statevector', 'set_density_matrix',
        'set_stabilizer'
    ])

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
        'basis_gates': _DEFAULT_BASIS_GATES,
        'custom_instructions': _DEFAULT_CUSTOM_INSTR,
        'gates': []
    }

    _SIMULATION_METHODS = [
        'automatic', 'statevector', 'statevector_gpu',
        'statevector_thrust', 'density_matrix',
        'density_matrix_gpu', 'density_matrix_thrust',
        'stabilizer', 'matrix_product_state', 'extended_stabilizer'
    ]

    _AVAILABLE_METHODS = None

    def __init__(self,
                 configuration=None,
                 properties=None,
                 provider=None,
                 **backend_options):

        warn('The `QasmSimulator` backend will be deprecated in the'
             ' future. It has been superseded by the `AerSimulator`'
             ' backend.', PendingDeprecationWarning)

        self._controller = qasm_controller_execute()

        # Update available methods for class
        if QasmSimulator._AVAILABLE_METHODS is None:
            QasmSimulator._AVAILABLE_METHODS = available_methods(
                self._controller, QasmSimulator._SIMULATION_METHODS)

        # Default configuration
        if configuration is None:
            configuration = QasmBackendConfiguration.from_dict(
                QasmSimulator._DEFAULT_CONFIGURATION)
        else:
            configuration.open_pulse = False

        # Cache basis gates since computing the intersection
        # of noise model, method, and config gates is expensive.
        self._cached_basis_gates = self._DEFAULT_BASIS_GATES

        super().__init__(configuration,
                         properties=properties,
                         available_methods=QasmSimulator._AVAILABLE_METHODS,
                         provider=provider,
                         backend_options=backend_options)

    def __repr__(self):
        """String representation of an AerBackend."""
        display = super().__repr__()[:-1]
        pad = ' ' * (len(self.__class__.__name__) + 1)

        method = getattr(self.options, 'method', None)
        if method not in [None, 'automatic']:
            display += ",\n{}method='{}'".format(pad, method)

        noise_model = getattr(self.options, 'noise_model', None)
        if noise_model is not None and not noise_model.is_ideal():
            display += ',\n{}noise_model={})'.format(pad, repr(noise_model))

        display += ")"
        return display

    @classmethod
    def _default_options(cls):
        return Options(
            # Global options
            shots=1024,
            method=None,
            precision="double",
            zero_threshold=1e-10,
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
            accept_distributed_results=None,
            blocking_qubits=None,
            blocking_enable=False,
            memory=None,
            noise_model=None,
            seed_simulator=None,
            # statevector options
            statevector_parallel_threshold=14,
            statevector_sample_measure_opt=10,
            # stabilizer options
            stabilizer_max_snapshot_probabilities=32,
            # extended stabilizer options
            extended_stabilizer_sampling_method='resampled_metropolis',
            extended_stabilizer_metropolis_mixing_time=5000,
            extended_stabilizer_approximation_error=0.05,
            extended_stabilizer_norm_estimation_samples=100,
            extended_stabilizer_norm_estimation_repitions=3,
            extended_stabilizer_parallel_threshold=100,
            extended_stabilizer_probabilities_snapshot_samples=3000,
            # MPS options
            matrix_product_state_truncation_threshold=1e-16,
            matrix_product_state_max_bond_dimension=None,
            mps_sample_measure_algorithm='mps_heuristic',
            chop_threshold=1e-8,
            mps_parallel_threshold=14,
            mps_omp_threads=1)

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

    def configuration(self):
        """Return the simulator backend configuration.

        Returns:
            BackendConfiguration: the configuration for the backend.
        """
        config = copy.copy(self._configuration)
        for key, val in self._options_configuration.items():
            setattr(config, key, val)
        # Update basis gates based on custom options, config, method,
        # and noise model
        config.custom_instructions = self._custom_instructions()
        config.basis_gates = self._cached_basis_gates + config.custom_instructions
        return config

    def _execute(self, qobj):
        """Execute a qobj on the backend.

        Args:
            qobj (QasmQobj): simulator input.

        Returns:
            dict: return a dictionary of results.
        """
        return cpp_execute(self._controller, qobj)

    def set_options(self, **fields):
        out_options = {}
        update_basis_gates = False
        for key, value in fields.items():
            if key == 'method':
                self._set_method_config(value)
                update_basis_gates = True
                out_options[key] = value
            elif key in ['noise_model', 'basis_gates']:
                update_basis_gates = True
                out_options[key] = value
            elif key == 'custom_instructions':
                self._set_configuration_option(key, value)
            else:
                out_options[key] = value
        super().set_options(**out_options)
        if update_basis_gates:
            self._cached_basis_gates = self._basis_gates()

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

    def _basis_gates(self):
        """Return simualtor basis gates.

        This will be the option value of basis gates if it was set,
        otherwise it will be the intersection of the configuration, noise model
        and method supported basis gates.
        """
        # Use option value for basis gates if set
        if 'basis_gates' in self._options_configuration:
            return self._options_configuration['basis_gates']

        # Compute intersection with method basis gates
        method_gates = self._method_basis_gates()
        config_gates = self._configuration.basis_gates
        if config_gates:
            basis_gates = set(config_gates).intersection(
                method_gates)
        else:
            basis_gates = method_gates

        # Compute intersection with noise model basis gates
        noise_model = getattr(self.options, 'noise_model', None)
        if noise_model:
            noise_gates = noise_model.basis_gates
            basis_gates = basis_gates.intersection(noise_gates)
        else:
            noise_gates = None

        if not basis_gates:
            logger.warning(
                "The intersection of configuration basis gates (%s), "
                "simulation method basis gates (%s), and "
                "noise model basis gates (%s) is empty",
                config_gates, method_gates, noise_gates)
        return sorted(basis_gates)

    def _method_basis_gates(self):
        """Return method basis gates and custom instructions"""
        method = self._options.get('method', None)
        if method in ['density_matrix', 'density_matrix_gpu', 'density_matrix_thrust']:
            return sorted([
                'u1', 'u2', 'u3', 'u', 'p', 'r', 'rx', 'ry', 'rz', 'id', 'x',
                'y', 'z', 'h', 's', 'sdg', 'sx', 't', 'tdg', 'swap', 'cx',
                'cy', 'cz', 'cp', 'cu1', 'rxx', 'ryy', 'rzz', 'rzx', 'ccx',
                'unitary', 'diagonal', 'delay', 'pauli'
            ])
        if method == 'matrix_product_state':
            return sorted([
                'u1', 'u2', 'u3', 'u', 'p', 'cp', 'cx', 'cy', 'cz', 'id', 'x', 'y', 'z', 'h', 's',
                'sdg', 'sx', 't', 'tdg', 'swap', 'ccx', 'unitary', 'roerror', 'delay',
                'r', 'rx', 'ry', 'rz', 'rxx', 'ryy', 'rzz', 'rzx', 'csx', 'cswap', 'diagonal',
                'initialize'
            ])
        if method == 'stabilizer':
            return sorted([
                'id', 'x', 'y', 'z', 'h', 's', 'sdg', 'sx', 'cx', 'cy', 'cz',
                'swap', 'delay',
            ])
        if method == 'extended_stabilizer':
            return sorted([
                'cx', 'cz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 'sx',
                'swap', 'u0', 't', 'tdg', 'u1', 'p', 'ccx', 'ccz', 'delay'
            ])
        return QasmSimulator._DEFAULT_BASIS_GATES

    def _custom_instructions(self):
        """Return method basis gates and custom instructions"""
        # pylint: disable = too-many-return-statements
        if 'custom_instructions' in self._options_configuration:
            return self._options_configuration['custom_instructions']

        method = self._options.get('method', None)
        if method in ['statevector', 'statevector_gpu', 'statevector_thrust']:
            return sorted([
                'roerror', 'kraus', 'snapshot', 'save_expval', 'save_expval_var',
                'save_probabilities', 'save_probabilities_dict',
                'save_amplitudes', 'save_amplitudes_sq', 'save_state',
                'save_density_matrix', 'save_statevector', 'save_statevector_dict',
                'set_statevector'
            ])
        if method in ['density_matrix', 'density_matrix_gpu', 'density_matrix_thrust']:
            return sorted([
                'roerror', 'kraus', 'superop', 'snapshot', 'save_expval', 'save_expval_var',
                'save_probabilities', 'save_probabilities_dict',
                'save_state', 'save_density_matrix', 'save_amplitudes_sq',
                'set_statevector', 'set_density_matrix'
            ])
        if method == 'matrix_product_state':
            return sorted([
                'roerror', 'snapshot', 'kraus', 'save_expval', 'save_expval_var',
                'save_probabilities', 'save_probabilities_dict',
                'save_density_matrix', 'save_state', 'save_statevector',
                'save_amplitudes', 'save_amplitudes_sq', 'save_matrix_product_state',
                'set_matrix_product_state'])
        if method == 'stabilizer':
            return sorted([
                'roerror', 'snapshot', 'save_expval', 'save_expval_var',
                'save_probabilities', 'save_probabilities_dict',
                'save_amplitudes_sq', 'save_state', 'save_stabilizer',
                'set_stabilizer'
            ])
        if method == 'extended_stabilizer':
            return sorted(['roerror', 'snapshot', 'save_statevector'])
        return QasmSimulator._DEFAULT_CUSTOM_INSTR

    def _set_method_config(self, method=None):
        """Set non-basis gate options when setting method"""
        super().set_options(method=method)
        # Update configuration description and number of qubits
        if method in ['statevector', 'statevector_gpu', 'statevector_thrust']:
            description = 'A C++ statevector simulator with noise'
            n_qubits = MAX_QUBITS_STATEVECTOR
        elif method in ['density_matrix', 'density_matrix_gpu', 'density_matrix_thrust']:
            description = 'A C++ density matrix simulator with noise'
            n_qubits = MAX_QUBITS_STATEVECTOR // 2
        elif method == 'matrix_product_state':
            description = 'A C++ matrix product state simulator with noise'
            n_qubits = 63  # TODO: not sure what to put here?
        elif method == 'stabilizer':
            description = 'A C++ Clifford stabilizer simulator with noise'
            n_qubits = 10000  # TODO: estimate from memory
        elif method == 'extended_stabilizer':
            description = 'A C++ Clifford+T extended stabilizer simulator with noise'
            n_qubits = 63  # TODO: estimate from memory
        else:
            # Clear options to default
            description = None
            n_qubits = None
        self._set_configuration_option('description', description)
        self._set_configuration_option('n_qubits', n_qubits)
