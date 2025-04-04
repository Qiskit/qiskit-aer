# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Aer qasm simulator backend.
"""

import copy
import logging
from qiskit.providers.options import Options
from qiskit.providers.backend import BackendV2

from ..version import __version__
from .aerbackend import AerBackend, AerError
from .backendconfiguration import AerBackendConfiguration
from .backendproperties import target_to_backend_properties
from .backend_utils import (
    cpp_execute_circuits,
    available_methods,
    available_devices,
    MAX_QUBITS_STATEVECTOR,
    BASIS_GATES,
)

# pylint: disable=import-error, no-name-in-module, abstract-method
from .controller_wrappers import aer_controller_execute

logger = logging.getLogger(__name__)


class AerSimulator(AerBackend):
    """
    Noisy quantum circuit simulator backend.

    **Configurable Options**

    The `AerSimulator` supports multiple simulation methods and
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
        backend = AerSimulator(method='density_matrix',
                                noise_model=noise_model)

    **Simulating an IBM Quantum Backend**

    The simulator can be automatically configured to mimic an IBM Quantum backend using
    the :meth:`from_backend` method. This will configure the simulator to use the
    basic device :class:`NoiseModel` for that backend, and the same basis gates
    and coupling map.

    .. code-block:: python

        backend = AerSimulator.from_backend(backend)

    **Returning the Final State**

    The final state of the simulator can be saved to the returned
    ``Result`` object by appending the
    :func:`~qiskit_aer.library.save_state` instruction to a
    quantum circuit. The format of the final state will depend on the
    simulation method used. Additional simulation data may also be saved
    using the other save instructions in :mod:`qiskit.provider.aer.library`.

    **Simulation Method Option**

    The simulation method is set using the ``method`` kwarg. A list supported
    simulation methods can be returned using :meth:`available_methods`, these
    are

    * ``"automatic"``: Default simulation method. Select the simulation
      method automatically based on the circuit and noise model.

    * ``"statevector"``: A dense statevector simulation that can sample
      measurement outcomes from *ideal* circuits with all measurements at
      end of the circuit. For noisy simulations each shot samples a
      randomly sampled noisy circuit from the noise model.

    * ``"density_matrix"``: A dense density matrix simulation that may
      sample measurement outcomes from *noisy* circuits with all
      measurements at end of the circuit.

    * ``"stabilizer"``: An efficient Clifford stabilizer state simulator
      that can simulate noisy Clifford circuits if all errors in the noise
      model are also Clifford errors.

    * ``"extended_stabilizer"``: An approximate simulated for Clifford + T
      circuits based on a state decomposition into ranked-stabilizer state.
      The number of terms grows with the number of non-Clifford (T) gates.

    * ``"matrix_product_state"``: A tensor-network statevector simulator that
      uses a Matrix Product State (MPS) representation for the state. This
      can be done either with or without truncation of the MPS bond dimensions
      depending on the simulator options. The default behaviour is no
      truncation.

    * ``"unitary"``: A dense unitary matrix simulation of an ideal circuit.
      This simulates the unitary matrix of the circuit itself rather than
      the evolution of an initial quantum state. This method can only
      simulate gates, it does not support measurement, reset, or noise.

    * ``"superop"``: A dense superoperator matrix simulation of an ideal or
      noisy circuit. This simulates the superoperator matrix of the circuit
      itself rather than the evolution of an initial quantum state. This method
      can simulate ideal and noisy gates, and reset, but does not support
      measurement.

    * ``"tensor_network"``: A tensor-network based simulation that supports
      both statevector and density matrix. Currently there is only available
      for GPU and accelerated by using cuTensorNet APIs of cuQuantum.

    **GPU Simulation**

    By default all simulation methods run on the CPU, however select methods
    also support running on a GPU if qiskit-aer was installed with GPU support
    on a compatible NVidia GPU and CUDA version.

    +--------------------------+---------------+
    | Method                   | GPU Supported |
    +==========================+===============+
    | ``automatic``            | Sometimes     |
    +--------------------------+---------------+
    | ``statevector``          | Yes           |
    +--------------------------+---------------+
    | ``density_matrix``       | Yes           |
    +--------------------------+---------------+
    | ``stabilizer``           | No            |
    +--------------------------+---------------+
    | ``matrix_product_state`` | No            |
    +--------------------------+---------------+
    | ``extended_stabilizer``  | No            |
    +--------------------------+---------------+
    | ``unitary``              | Yes           |
    +--------------------------+---------------+
    | ``superop``              | No            |
    +--------------------------+---------------+
    | ``tensor_network``       | Yes(GPU only) |
    +--------------------------+---------------+

    Running a GPU simulation is done using ``device="GPU"`` kwarg during
    initialization or with :meth:`set_options`. The list of supported devices
    for the current system can be returned using :meth:`available_devices`.

    For multiple shots simulation, OpenMP threads should be exploited for
    multi-GPUs. Number of GPUs used for multi-shots is reported in
    metadata ``gpu_parallel_shots_`` or is batched execution is done reported
    in metadata ``batched_shots_optimization_parallel_gpus``.
    For large qubits circuits with multiple GPUs, number of GPUs is reported
    in metadata ``chunk_parallel_gpus`` in ``cacheblocking``.

    If AerSimulator is built with cuStateVec support, cuStateVec APIs are enabled
    by setting ``cuStateVec_enable=True``.

    * ``target_gpus`` (list): List of GPU's IDs starting from 0 sets
      the target GPUs used for the simulation.
      If this option is not specified, all the available GPUs are used for
      chunks/shots distribution.

    **Additional Backend Options**

    The following simulator specific backend options are supported

    * ``method`` (str): Set the simulation method (Default: ``"automatic"``).
      Use :meth:`available_methods` to return a list of all availabe methods.

    * ``device`` (str): Set the simulation device (Default: ``"CPU"``).
      Use :meth:`available_devices` to return a list of devices supported
      on the current system.

    * ``precision`` (str): Set the floating point precision for
      certain simulation methods to either ``"single"`` or ``"double"``
      precision (default: ``"double"``).

    * ``executor`` (futures.Executor or None): Set a custom executor for
      asynchronous running of simulation jobs (Default: None).

    * ``max_job_size`` (int or None): If the number of run circuits
      exceeds this value simulation will be run as a set of of sub-jobs
      on the executor. If ``None`` simulation of all circuits are submitted
      to the executor as a single job (Default: None).

    * ``max_shot_size`` (int or None): If the number of shots of a noisy
      circuit exceeds this value simulation will be split into multi
      circuits for execution and the results accumulated. If ``None``
      circuits will not be split based on shots. When splitting circuits
      use the ``max_job_size`` option to control how these split circuits
      should be submitted to the executor (Default: None).

      a noise model exceeds this value simulation will be splitted into
      sub-circuits. If ``None``  simulator does noting (Default: None).

    * ``enable_truncation`` (bool): If set to True this removes unnecessary
      qubits which do not affect the simulation outcome from the simulated
      circuits (Default: True).

    * ``zero_threshold`` (double): Sets the threshold for truncating
      small values to zero in the result data (Default: 1e-10).

    * ``validation_threshold`` (double): Sets the threshold for checking
      if initial states are valid (Default: 1e-8).

    * ``max_parallel_threads`` (int): Sets the maximum number of CPU
      cores used by OpenMP for parallelization. If set to 0 the
      maximum will be set to the number of CPU cores (Default: 0).

    * ``max_parallel_experiments`` (int): Sets the maximum number of
      experiments that may be executed in parallel up to the
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
      to store quantum states. If quantum states need more, an error
      is thrown unless -1 is set. In general, a state vector of n-qubits
      uses 2^n complex values (16 Bytes).
      If set to 0, the maximum will be automatically set to
      the system memory size (Default: 0).

    * ``cuStateVec_enable`` (bool): This option enables accelerating by
      cuStateVec library of cuQuantum from NVIDIA, that has highly optimized
      kernels for GPUs (Default: False). This option will be ignored
      if AerSimulator is not built with cuStateVec support.

    * ``blocking_enable`` (bool): This option enables parallelization with
      multiple GPUs or multiple processes with MPI (CPU/GPU). This option
      is only available for ``"statevector"``, ``"density_matrix"`` and
      ``"unitary"`` (Default: False).

    * ``blocking_qubits`` (int): Sets the number of qubits of chunk size
      used for parallelizing with multiple GPUs or multiple processes with
      MPI (CPU/GPU). 16*2^blocking_qubits should be less than 1/4 of the GPU
      memory in double precision. This option is only available for
      ``"statevector"``, ``"density_matrix"`` and ``"unitary"``.
      This option should be set when using option ``blocking_enable=True``
      (Default: 0).
      If multiple GPUs are used for parallelization number of GPUs is
      reported to ``chunk_parallel_gpus`` in ``cacheblocking`` metadata.

    * ``chunk_swap_buffer_qubits`` (int): Sets the number of qubits of
      maximum buffer size (=2^chunk_swap_buffer_qubits) used for multiple
      chunk-swaps over MPI processes. This parameter should be smaller than
      ``blocking_qubits`` otherwise multiple chunk-swaps is disabled.
      ``blocking_qubits`` - ``chunk_swap_buffer_qubits`` swaps are applied
      at single all-to-all communication. (Default: 15).

    * ``batched_shots_gpu`` (bool): This option enables batched execution
      of multiple shot simulations on GPU devices for GPU enabled simulation
      methods. This optimization is intended for statevector simulations with
      noise models, or statevecor and density matrix simulations with
      intermediate measurements and can greatly accelerate simulation time
      on GPUs. If there are multiple GPUs on the system, shots are distributed
      automatically across available GPUs. Also this option distributes multiple
      shots to parallel processes of MPI (Default: False).
      If multiple GPUs are used for batched exectuion number of GPUs is
      reported to ``batched_shots_optimization_parallel_gpus`` metadata.
      ``cuStateVec_enable`` is not supported for this option.

    * ``batched_shots_gpu_max_qubits`` (int): This option sets the maximum
      number of qubits for enabling the ``batched_shots_gpu`` option. If the
      number of active circuit qubits is greater than this value batching of
      simulation shots will not be used. (Default: 16).

    * ``num_threads_per_device`` (int): This option sets the number of
      threads per device. For GPU simulation, this value sets number of
      threads per GPU. This parameter is used to optimize Pauli noise
      simulation with multiple-GPUs (Default: 1).

    * ``shot_branching_enable`` (bool): This option enables/disables
      applying shot-branching technique to speed up multi-shots of dynamic
      circutis simulations or circuits simulations with noise models.
      (Default: False).
      Starting from single state shared with multiple shots and
      state will be branched dynamically at runtime.
      This option can decrease runs of shots if there will be less branches
      than number of total shots.
      This option is available for ``"statevector"``, ``"density_matrix"``
      and ``"tensor_network"``.
      WARNING: `shot_branching` option is unstable on MacOS currently

    * ``shot_branching_sampling_enable`` (bool): This option enables/disables
      applying sampling measure if the input circuit has all the measure
      operations at the end of the circuit. (Default: False).
      Because measure operation branches state into 2 states, it is not
      efficient to apply branching for measure.
      Sampling measure improves speed to get counts for multiple-shots
      sharing the same state.
      Note that the counts obtained by sampling measure may not be as same as
      the counts calculated by multiple measure operations,
      becuase sampling measure takes only one randome number per shot.
      This option is available for ``"statevector"``, ``"density_matrix"``
      and ``"tensor_network"``.

    * ``accept_distributed_results`` (bool): This option enables storing
      results independently in each process (Default: None).

    * ``runtime_parameter_bind_enable`` (bool): If this option is True
      parameters are bound at runtime by using multi-shots without constructing
      circuits for each parameters. For GPU this option can be used with
      ``batched_shots_gpu`` to run with multiple parameters in a batch.
      (Default: False).

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
      qubit number for the :class:`~qiskit_aer.library.SaveProbabilities` instruction (Default: 32).

    These backend options only apply when using the ``"extended_stabilizer"``
    simulation method:

    * ``extended_stabilizer_sampling_method`` (string): Choose how to simulate
      measurements on qubits. The performance of the simulator depends
      significantly on this choice. In the following, let n be the number of
      qubits in the circuit, m the number of qubits measured, and S be the
      number of shots (Default: resampled_metropolis).

      - ``"metropolis"``: Use a Monte-Carlo method to sample many output
        strings from the simulator at once. To be accurate, this method
        requires that all the possible output strings have a non-zero
        probability. It will give inaccurate results on cases where
        the circuit has many zero-probability outcomes.
        This method has an overall runtime that scales as n^{2} + (S-1)n.

      - ``"resampled_metropolis"``: A variant of the metropolis method,
        where the Monte-Carlo method is reinitialised for every shot. This
        gives better results for circuits where some outcomes have zero
        probability, but will still fail if the output distribution
        is sparse. The overall runtime scales as Sn^{2}.

      - ``"norm_estimation"``: An alternative sampling method using
        random state inner products to estimate outcome probabilites. This
        method requires twice as much memory, and significantly longer
        runtimes, but gives accurate results on circuits with sparse
        output distributions. The overall runtime scales as Sn^{3}m^{3}.

    * ``extended_stabilizer_metropolis_mixing_time`` (int): Set how long the
      monte-carlo method runs before performing measurements. If the
      output distribution is strongly peaked, this can be decreased
      alongside setting extended_stabilizer_disable_measurement_opt
      to True (Default: 5000).

    * ``extended_stabilizer_approximation_error`` (double): Set the error
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

    These backend options only apply when using the ``matrix_product_state``
    simulation method:

    * ``matrix_product_state_max_bond_dimension`` (int): Sets a limit
      on the number of Schmidt coefficients retained at the end of
      the svd algorithm. Coefficients beyond this limit will be discarded.
      (Default: None, i.e., no limit on the bond dimension).

    * ``matrix_product_state_truncation_threshold`` (double):
      Discard the smallest coefficients for which the sum of
      their squares is smaller than this threshold.
      (Default: 1e-16).

    * ``mps_sample_measure_algorithm`` (str): Choose which algorithm to use for
      ``"sample_measure"`` (Default: "mps_apply_measure").

      - ``mps_probabilities``: This method first constructs the probability
        vector and then generates a sample per shot. It is more efficient for
        a large number of shots and a small number of qubits, with complexity
        O(2^n * n * D^2) to create the vector and O(1) per shot, where n is
        the number of qubits and D is the bond dimension.

      - ``mps_apply_measure``: This method creates a copy of the mps structure
        and measures directly on it. It is more efficient for a small number of
        shots, and a large number of qubits, with complexity around
        O(n * D^2) per shot.

    * ``mps_log_data`` (bool): if True, output logging data of the MPS
      structure: bond dimensions and values discarded during approximation.
      (Default: False)

    * ``mps_swap_direction`` (str): Determine the direction of swapping the
      qubits when internal swaps are inserted for a 2-qubit gate.
      Possible values are "mps_swap_right" and "mps_swap_left".
      (Default: "mps_swap_left")

    * ``chop_threshold`` (float): This option sets a threshold for
      truncating snapshots (Default: 1e-8).

    * ``mps_parallel_threshold`` (int): This option sets OMP number threshold (Default: 14).

    * ``mps_omp_threads`` (int): This option sets the number of OMP threads (Default: 1).

    * ``mps_lapack`` (bool): This option indicates to compute the SVD function
      using OpenBLAS/Lapack interface (Default: False).

    These backend options only apply when using the ``tensor_network``
    simulation method:

    * ``tensor_network_num_sampling_qubits`` (int): is used to set number
      of qubits to be sampled in single tensor network contraction when
      using sampling measure. (Default: 10)

    * ``use_cuTensorNet_autotuning`` (bool): enables auto tuning of plan
      in cuTensorNet API. It takes some time for tuning, so enable if the
      circuit is very large. (Default: False)

    These backend options apply in circuit optimization passes:

    * ``fusion_enable`` (bool): Enable fusion optimization in circuit
      optimization passes [Default: True]
    * ``fusion_verbose`` (bool): Output gates generated in fusion optimization
      into metadata [Default: False]
    * ``fusion_max_qubit`` (int): Maximum number of qubits for a operation generated
      in a fusion optimization. A default value (``None``) automatically sets a value
      depending on the simulation method: [Default: None]
    * ``fusion_threshold`` (int): Threshold that number of qubits must be greater
      than or equal to enable fusion optimization. A default value automatically sets
      a value depending on the simulation method [Default: None]

    ``fusion_enable`` and ``fusion_threshold`` are set as follows if their default
    values (``None``) are configured:

    +--------------------------+----------------------+----------------------+
    | Method                   | ``fusion_max_qubit`` | ``fusion_threshold`` |
    +==========================+======================+======================+
    | ``statevector``          | 5                    | 14                   |
    +--------------------------+----------------------+----------------------+
    | ``density_matrix``       | 2                    | 7                    |
    +--------------------------+----------------------+----------------------+
    | ``unitary``              | 5                    | 7                    |
    +--------------------------+----------------------+----------------------+
    | ``superop``              | 2                    | 7                    |
    +--------------------------+----------------------+----------------------+
    | other methods            | 5                    | 14                   |
    +--------------------------+----------------------+----------------------+

    """

    _BASIS_GATES = BASIS_GATES

    _CUSTOM_INSTR = {
        "statevector": sorted(
            [
                "quantum_channel",
                "qerror_loc",
                "roerror",
                "kraus",
                "save_expval",
                "save_expval_var",
                "save_probabilities",
                "save_probabilities_dict",
                "save_amplitudes",
                "save_amplitudes_sq",
                "save_density_matrix",
                "save_state",
                "save_statevector",
                "save_statevector_dict",
                "set_statevector",
                "if_else",
                "for_loop",
                "while_loop",
                "break_loop",
                "continue_loop",
                "initialize",
                "reset",
                "switch_case",
                "delay",
            ]
        ),
        "density_matrix": sorted(
            [
                "quantum_channel",
                "qerror_loc",
                "roerror",
                "kraus",
                "superop",
                "save_state",
                "save_expval",
                "save_expval_var",
                "save_probabilities",
                "save_probabilities_dict",
                "save_density_matrix",
                "save_amplitudes_sq",
                "set_density_matrix",
                "if_else",
                "for_loop",
                "while_loop",
                "break_loop",
                "continue_loop",
                "reset",
                "switch_case",
                "delay",
            ]
        ),
        "matrix_product_state": sorted(
            [
                "quantum_channel",
                "qerror_loc",
                "roerror",
                "kraus",
                "save_expval",
                "save_expval_var",
                "save_probabilities",
                "save_probabilities_dict",
                "save_state",
                "save_matrix_product_state",
                "save_statevector",
                "save_density_matrix",
                "save_amplitudes",
                "save_amplitudes_sq",
                "set_matrix_product_state",
                "if_else",
                "for_loop",
                "while_loop",
                "break_loop",
                "continue_loop",
                "initialize",
                "reset",
                "switch_case",
                "delay",
            ]
        ),
        "stabilizer": sorted(
            [
                "quantum_channel",
                "qerror_loc",
                "roerror",
                "save_expval",
                "save_expval_var",
                "save_probabilities",
                "save_probabilities_dict",
                "save_amplitudes_sq",
                "save_state",
                "save_clifford",
                "save_stabilizer",
                "set_stabilizer",
                "if_else",
                "for_loop",
                "while_loop",
                "break_loop",
                "continue_loop",
                "reset",
                "switch_case",
                "delay",
            ]
        ),
        "extended_stabilizer": sorted(
            [
                "quantum_channel",
                "qerror_loc",
                "roerror",
                "save_statevector",
                "reset",
                "delay",
            ]
        ),
        "unitary": sorted(
            [
                "save_state",
                "save_unitary",
                "set_unitary",
                "reset",
                "delay",
            ]
        ),
        "superop": sorted(
            [
                "quantum_channel",
                "qerror_loc",
                "kraus",
                "superop",
                "save_state",
                "save_superop",
                "set_superop",
                "reset",
                "delay",
            ]
        ),
        "tensor_network": sorted(
            [
                "quantum_channel",
                "qerror_loc",
                "roerror",
                "kraus",
                "superop",
                "save_state",
                "save_expval",
                "save_expval_var",
                "save_probabilities",
                "save_probabilities_dict",
                "save_density_matrix",
                "save_amplitudes",
                "save_amplitudes_sq",
                "save_statevector",
                "save_statevector_dict",
                "set_statevector",
                "set_density_matrix",
                "initialize",
                "reset",
                "switch_case",
                "delay",
            ]
        ),
    }

    # Automatic method custom instructions are the union of statevector,
    # density matrix, and stabilizer methods
    _CUSTOM_INSTR[None] = _CUSTOM_INSTR["automatic"] = sorted(
        set(_CUSTOM_INSTR["statevector"])
        .union(_CUSTOM_INSTR["stabilizer"])
        .union(_CUSTOM_INSTR["density_matrix"])
        .union(_CUSTOM_INSTR["matrix_product_state"])
        .union(_CUSTOM_INSTR["unitary"])
        .union(_CUSTOM_INSTR["superop"])
        .union(_CUSTOM_INSTR["tensor_network"])
    )

    _DEFAULT_CONFIGURATION = {
        "backend_name": "aer_simulator",
        "backend_version": __version__,
        "n_qubits": MAX_QUBITS_STATEVECTOR,
        "url": "https://github.com/Qiskit/qiskit-aer",
        "simulator": True,
        "local": True,
        "conditional": True,
        "memory": True,
        "max_shots": int(1e6),
        "description": "A C++ Qasm simulator with noise",
        "coupling_map": None,
        "basis_gates": BASIS_GATES["automatic"],
        "custom_instructions": _CUSTOM_INSTR["automatic"],
        "gates": [],
    }

    _SIMULATION_METHODS = [
        "automatic",
        "statevector",
        "density_matrix",
        "stabilizer",
        "matrix_product_state",
        "extended_stabilizer",
        "unitary",
        "superop",
        "tensor_network",
    ]

    _AVAILABLE_METHODS = None

    _SIMULATION_DEVICES = ("CPU", "GPU", "Thrust")

    _AVAILABLE_DEVICES = None

    def __init__(
        self, configuration=None, properties=None, provider=None, target=None, **backend_options
    ):
        self._controller = aer_controller_execute()

        # Update available methods and devices for class
        if AerSimulator._AVAILABLE_DEVICES is None:
            AerSimulator._AVAILABLE_DEVICES = available_devices(self._controller)
        if AerSimulator._AVAILABLE_METHODS is None:
            AerSimulator._AVAILABLE_METHODS = available_methods(
                AerSimulator._SIMULATION_METHODS, AerSimulator._AVAILABLE_DEVICES
            )

        # Default configuration
        if configuration is None:
            configuration = AerBackendConfiguration.from_dict(AerSimulator._DEFAULT_CONFIGURATION)

        # set backend name from method and device in option
        if "from" not in configuration.backend_name:
            method = "automatic"
            device = "CPU"
            for key, value in backend_options.items():
                if key == "method":
                    method = value
                if key == "device":
                    device = value
            if method not in [None, "automatic"]:
                configuration.backend_name += f"_{method}"
            if device not in [None, "CPU"]:
                configuration.backend_name += f"_{device}".lower()

        # Cache basis gates since computing the intersection
        # of noise model, method, and config gates is expensive.
        self._cached_basis_gates = self._BASIS_GATES["automatic"]

        super().__init__(
            configuration,
            properties=properties,
            provider=provider,
            target=target,
            backend_options=backend_options,
        )

        if "basis_gates" in backend_options.items():
            self._check_basis_gates(backend_options["basis_gates"])

    @classmethod
    def _default_options(cls):
        return Options(
            # Global options
            shots=1024,
            method="automatic",
            device="CPU",
            precision="double",
            executor=None,
            max_job_size=None,
            max_shot_size=None,
            enable_truncation=True,
            zero_threshold=1e-10,
            validation_threshold=None,
            max_parallel_threads=None,
            max_parallel_experiments=None,
            max_parallel_shots=None,
            max_memory_mb=None,
            fusion_enable=True,
            fusion_verbose=False,
            fusion_max_qubit=None,
            fusion_threshold=None,
            accept_distributed_results=None,
            memory=None,
            noise_model=None,
            seed_simulator=None,
            # cuStateVec (cuQuantum) option
            cuStateVec_enable=False,
            # cache blocking for multi-GPUs/MPI options
            blocking_qubits=None,
            blocking_enable=False,
            chunk_swap_buffer_qubits=None,
            # multi-shots optimization options (GPU only)
            batched_shots_gpu=False,
            batched_shots_gpu_max_qubits=16,
            num_threads_per_device=1,
            # multi-shot branching
            shot_branching_enable=False,
            shot_branching_sampling_enable=False,
            # statevector options
            statevector_parallel_threshold=14,
            statevector_sample_measure_opt=10,
            # stabilizer options
            stabilizer_max_snapshot_probabilities=32,
            # extended stabilizer options
            extended_stabilizer_sampling_method="resampled_metropolis",
            extended_stabilizer_metropolis_mixing_time=5000,
            extended_stabilizer_approximation_error=0.05,
            extended_stabilizer_norm_estimation_samples=100,
            extended_stabilizer_norm_estimation_repetitions=3,
            extended_stabilizer_parallel_threshold=100,
            extended_stabilizer_probabilities_snapshot_samples=3000,
            # MPS options
            matrix_product_state_truncation_threshold=1e-16,
            matrix_product_state_max_bond_dimension=None,
            mps_sample_measure_algorithm="mps_heuristic",
            mps_log_data=False,
            mps_swap_direction="mps_swap_left",
            chop_threshold=1e-8,
            mps_parallel_threshold=14,
            mps_omp_threads=1,
            mps_lapack=False,
            # tensor network options
            tensor_network_num_sampling_qubits=10,
            use_cuTensorNet_autotuning=False,
            # parameter binding
            runtime_parameter_bind_enable=False,
        )

    def __repr__(self):
        """String representation of an AerSimulator."""
        display = super().__repr__()
        noise_model = getattr(self.options, "noise_model", None)
        if noise_model is None or noise_model.is_ideal():
            return display
        pad = " " * (len(self.__class__.__name__) + 1)
        return f"{display[:-1]}\n{pad}noise_model={repr(noise_model)})"

    @classmethod
    def from_backend(cls, backend, **options):
        """Initialize simulator from backend."""
        if isinstance(backend, BackendV2):
            if backend.description is None:
                description = "created by AerSimulator.from_backend"
            else:
                description = backend.description

            configuration = AerBackendConfiguration(
                backend_name=f"aer_simulator_from({backend.name})",
                backend_version=backend.backend_version,
                n_qubits=backend.num_qubits,
                basis_gates=backend.operation_names,
                gates=[],
                max_shots=int(1e6),
                coupling_map=list(backend.coupling_map.get_edges()),
                max_experiments=backend.max_circuits,
                description=description,
            )
            properties = target_to_backend_properties(backend.target)
            target = backend.target
        else:
            raise TypeError(
                "The backend argument requires a BackendV2 object, " f"not a {type(backend)} object"
            )
        # Use automatic noise model if none is provided
        if "noise_model" not in options:
            # pylint: disable=import-outside-toplevel
            # Avoid cyclic import
            from ..noise.noise_model import NoiseModel

            noise_model = NoiseModel.from_backend(backend)
            if not noise_model.is_ideal():
                options["noise_model"] = noise_model

        # Initialize simulator
        sim = cls(configuration=configuration, properties=properties, target=target, **options)
        return sim

    def available_methods(self):
        """Return the available simulation methods."""
        return copy.copy(self._AVAILABLE_METHODS)

    def available_devices(self):
        """Return the available simulation methods."""
        if "_gpu" in self.name:
            return ["GPU"]
        return copy.copy(self._AVAILABLE_DEVICES)

    def configuration(self):
        """Return the simulator backend configuration.

        Returns:
            BackendConfiguration: the configuration for the backend.
        """
        config = copy.copy(self._configuration)
        for key, val in self._options_configuration.items():
            setattr(config, key, val)

        method = getattr(self.options, "method", "automatic")

        # Update basis gates based on custom options, config, method,
        # and noise model
        config.custom_instructions = self._CUSTOM_INSTR[method]
        config.basis_gates = self._cached_basis_gates + config.custom_instructions
        return config

    def _execute_circuits(self, aer_circuits, noise_model, config):
        """Execute circuits on the backend."""
        ret = cpp_execute_circuits(self._controller, aer_circuits, noise_model, config)
        return ret

    def set_option(self, key, value):
        if key == "custom_instructions":
            self._set_configuration_option(key, value)
            return
        if key == "method":
            if value is not None and value not in self.available_methods():
                raise AerError(
                    f"Invalid simulation method {value}. Available methods"
                    f" are: {self.available_methods()}"
                )
            self._set_method_config(value)
        if key == "basis_gates":
            self._check_basis_gates(value)

        super().set_option(key, value)
        if key in ["method", "noise_model", "basis_gates"]:
            self._cached_basis_gates = self._basis_gates()

        # update backend name
        if key in ["method", "device"]:
            if "from" not in self.name:
                if key == "method":
                    self.name = "aer_simulator"
                    if value != "automatic":
                        self.name += f"_{value}"
                        device = getattr(self.options, "device", "CPU")
                        if device != "CPU":
                            self.name += f"_{device}".lower()
                if key == "device":
                    method = getattr(self.options, "method", "auto")
                    self.name = "aer_simulator"
                    if method != "automatic":
                        self.name += f"_{method}"
                        if value != "CPU":
                            self.name += f"_{value}".lower()

    def _basis_gates(self):
        """Return simualtor basis gates.

        This will be the option value of basis gates if it was set,
        otherwise it will be the intersection of the configuration, noise model
        and method supported basis gates.
        """
        # Use option value for basis gates if set
        if "basis_gates" in self._options_configuration:
            return self._options_configuration["basis_gates"]

        # Compute intersection with method basis gates
        method = getattr(self._options, "method", "automatic")
        method_gates = self._BASIS_GATES[method]
        config_gates = self._configuration.basis_gates
        if config_gates:
            basis_gates = set(config_gates).intersection(method_gates)
        else:
            basis_gates = method_gates

        # Compute intersection with noise model basis gates
        noise_model = getattr(self.options, "noise_model", None)
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
                config_gates,
                method_gates,
                noise_gates,
            )
        return sorted(basis_gates)

    def _set_method_config(self, method=None):
        """Set non-basis gate options when setting method"""
        # Update configuration description and number of qubits
        if method == "statevector":
            description = "A C++ statevector simulator with noise"
            n_qubits = MAX_QUBITS_STATEVECTOR
        elif method == "density_matrix":
            description = "A C++ density matrix simulator with noise"
            n_qubits = MAX_QUBITS_STATEVECTOR // 2
        elif method == "unitary":
            description = "A C++ unitary matrix simulator"
            n_qubits = MAX_QUBITS_STATEVECTOR // 2
        elif method == "superop":
            description = "A C++ superop matrix simulator with noise"
            n_qubits = MAX_QUBITS_STATEVECTOR // 4
        elif method == "matrix_product_state":
            description = "A C++ matrix product state simulator with noise"
            n_qubits = 63  # TODO: not sure what to put here?
        elif method == "stabilizer":
            description = "A C++ Clifford stabilizer simulator with noise"
            n_qubits = 10000  # TODO: estimate from memory
        elif method == "extended_stabilizer":
            description = "A C++ Clifford+T extended stabilizer simulator with noise"
            n_qubits = 63  # TODO: estimate from memory
        else:
            # Clear options to default
            description = None
            n_qubits = None

        if self._configuration.coupling_map:
            n_qubits = max(list(map(max, self._configuration.coupling_map))) + 1

        self._set_configuration_option("description", description)
        self._set_configuration_option("n_qubits", n_qubits)

    def _check_basis_gates(self, basis_gates):
        method = getattr(self.options, "method", "automatic")
        # check if basis_gates contains non-supported gates
        if method != "automatic":
            for gate in basis_gates:
                if gate not in self._BASIS_GATES[method]:
                    raise AerError(f"Invalid gate {gate} for simulation method {method}.")
