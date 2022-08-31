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
Noise model class for Qiskit Aer simulators.
"""
import copy
import json
import logging
from typing import Optional
from warnings import warn, catch_warnings, filterwarnings

from numpy import ndarray

from qiskit.circuit import Instruction, Delay
from qiskit.providers import QubitProperties
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.providers.models import BackendProperties
from qiskit.transpiler import PassManager
from qiskit.utils import apply_prefix
from .device.models import _excited_population, _truncate_t2_value
from .device.models import basic_device_gate_errors
from .device.models import basic_device_readout_errors
from .errors.quantum_error import QuantumError
from .errors.readout_error import ReadoutError
from .noiseerror import NoiseError
from .passes import RelaxationNoisePass
from ..backends.backend_utils import BASIS_GATES

logger = logging.getLogger(__name__)


class AerJSONEncoder(json.JSONEncoder):
    """
    JSON encoder for NumPy arrays and complex numbers.

    This functions as the standard JSON Encoder but adds support
    for encoding:
        complex numbers z as lists [z.real, z.imag]
        ndarrays as nested lists.
    """

    # pylint: disable=method-hidden,arguments-differ
    def default(self, obj):
        if isinstance(obj, ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


class QuantumErrorLocation(Instruction):
    """Instruction for referencing a multi-qubit error in a NoiseModel"""

    _directive = True

    def __init__(self, qerror):
        """Construct a new quantum error location instruction.

        Args:
            qerror (QuantumError): the quantum error to reference.
        """
        super().__init__("qerror_loc", qerror.num_qubits, 0, [], label=qerror.id)


class NoiseModel:
    """Noise model class for Qiskit Aer simulators.

    This class is used to represent noise model for the
    :class:`~qiskit_aer.QasmSimulator`. It can be used to construct
    custom noise models for simulator, or to automatically generate a basic
    device noise model for an IBMQ backend. See the
    :mod:`~qiskit_aer.noise` module documentation for additional
    information.

    **Example: Basic device noise model**

    An approximate :class:`NoiseModel` can be generated automatically from the
    properties of real device backends from the IBMQ provider using the
    :meth:`~NoiseModel.from_backend` method.

    .. code-block:: python

        from qiskit import IBMQ, Aer
        from qiskit_aer.noise import NoiseModel

        provider = IBMQ.load_account()
        backend = provider.get_backend('ibmq_vigo')
        noise_model = NoiseModel.from_backend(backend)
        print(noise_model)


    **Example: Custom noise model**

    Custom noise models can be used by adding :class:`QuantumError` to circuit
    gate, reset or measure instructions, and :class:`ReadoutError` to measure
    instructions.

    .. code-block:: python

        import qiskit_aer.noise as noise

        # Error probabilities
        prob_1 = 0.001  # 1-qubit gate
        prob_2 = 0.01   # 2-qubit gate

        # Depolarizing quantum errors
        error_1 = noise.depolarizing_error(prob_1, 1)
        error_2 = noise.depolarizing_error(prob_2, 2)

        # Add errors to noise model
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ['rz', 'sx', 'x'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
        print(noise_model)

    """
    # Checks for standard 1-3 qubit instructions
    _1qubit_instructions = set([
        'u1', 'u2', 'u3', 'u', 'p', 'r', 'rx', 'ry', 'rz', 'id', 'x',
        'y', 'z', 'h', 's', 'sdg', 'sx', 'sxdg', 't', 'tdg'])
    _2qubit_instructions = set([
        'swap', 'cx', 'cy', 'cz', 'csx', 'cp', 'cu', 'cu1', 'cu2', 'cu3', 'rxx',
        'ryy', 'rzz', 'rzx'])
    _3qubit_instructions = set(['ccx', 'cswap'])

    def __init__(self, basis_gates=None):
        """Initialize an empty noise model.

        Args:
            basis_gates (list[str] or None): Specify an initial basis_gates
                for the noise model. If None a default value of ['id', 'rz', 'sx', 'cx']
                is used (Default: None).

        Additional Information:
        Errors added to the noise model will have their instruction
        appended to the noise model basis_gates if the instruction is in
        the :class:`~qiskit_aer.QasmSimulator` basis_gates. If
        the instruction is not in the
        :class:`~qiskit_aer.QasmSimulator` basis_gates it is
        assumed to be a label for a standard gate, and that gate should be
        added to the `NoiseModel` basis_gates either using the init method,
        or the :meth:`add_basis_gates` method.
        """
        if basis_gates is None:
            # Default basis gates is id, rz, sx, cx so that all standard
            # non-identity instructions can be unrolled to rz, sx, cx,
            # and identities won't be unrolled
            self._basis_gates = set(['id', 'rz', 'sx', 'cx'])
        else:
            self._basis_gates = set(
                name for name, _ in self._instruction_names_labels(basis_gates))
        # Store gates with a noise model defined
        self._noise_instructions = set()
        # Store qubits referenced in noise model.
        # These include gate qubits in local quantum and readout errors,
        # and both gate and noise qubits for non-local quantum errors.
        self._noise_qubits = set()
        # Default (all-qubit) quantum errors are stored as:
        # dict(str: QuantumError)
        # where they keys are the instruction str label
        self._default_quantum_errors = {}
        # Local quantum errors are stored as:
        # dict(str: dict(tuple: QuantumError))
        # where the outer keys are the instruction str label and the
        # inner dict keys are the gate qubits
        self._local_quantum_errors = {}
        # Non-local quantum errors are stored as:
        # dict(str: dict(tuple: dict(tuple: QuantumError)))
        # where the outer keys are the instruction str label, the middle dict
        # keys are the gate qubits, and the inner most dict keys are
        # the noise qubits.
        self._nonlocal_quantum_errors = {}
        # Default (all-qubit) readout error is stored as a single
        # ReadoutError object since there may only be one defined.
        self._default_readout_error = None
        # Local readout errors are stored as:
        # dict(tuple: ReadoutError)
        # where the dict keys are the gate qubits.
        self._local_readout_errors = {}
        # Custom noise passes
        self._custom_noise_passes = []

    @property
    def basis_gates(self):
        """Return basis_gates for compiling to the noise model."""
        # Convert noise instructions to basis_gates string
        return sorted(self._basis_gates)

    @property
    def noise_instructions(self):
        """Return the set of noisy instructions for this noise model."""
        return sorted(self._noise_instructions)

    @property
    def noise_qubits(self):
        """Return the set of noisy qubits for this noise model."""
        return sorted(self._noise_qubits)

    @classmethod
    def from_backend(cls, backend,
                     gate_error=True,
                     readout_error=True,
                     thermal_relaxation=True,
                     temperature=0,
                     gate_lengths=None,
                     gate_length_units='ns',
                     standard_gates=None,
                     warnings=None):
        """Return a noise model derived from a devices backend properties.

        This function generates a noise model based on:

        * 1 and 2 qubit gate errors consisting of a
          :func:`depolarizing_error` followed
          by a :func:`thermal_relaxation_error`.

        * Single qubit :class:`ReadoutError` on all measurements.

        The Error error parameters are tuned for each individual qubit based on
        the :math:`T_1`, :math:`T_2`, frequency and readout error parameters for
        each qubit, and the gate error and gate time parameters for each gate
        obtained from the device backend properties.

        **Additional Information**

        The noise model includes the following errors:

        * If ``readout_error=True`` include single qubit readout
          errors on measurements.

        * If ``gate_error=True`` and ``thermal_relaxation=True`` include:

            * Single-qubit gate errors consisting of a :func:`depolarizing_error`
              followed by a :func:`thermal_relaxation_error` for the qubit the
              gate acts on.

            * Two-qubit gate errors consisting of a 2-qubit
              :func:`depolarizing_error` followed by single qubit
              :func:`thermal_relaxation_error` on each qubit participating in
              the gate.

        * If ``gate_error=True`` is ``True`` and ``thermal_relaxation=False``:

            * An N-qubit :func:`depolarizing_error` on each N-qubit gate.

        * If ``gate_error=False`` and ``thermal_relaxation=True`` include
          single-qubit :func:`thermal_relaxation_errors` on each qubits
          participating in a multi-qubit gate.

        For best practice in simulating a backend make sure that the
        circuit is compiled using the set of basis gates in the noise
        module by setting ``basis_gates=noise_model.basis_gates``
        and using the device coupling map with
        ``coupling_map=backend.configuration().coupling_map``

        **Specifying custom gate times**

        The ``gate_lengths`` kwarg can be used to specify custom gate times
        to add gate errors using the :math:`T_1` and :math:`T_2` values from
        the backend properties. This should be passed as a list of tuples
        ``gate_lengths=[(name, value), ...]``
        where ``name`` is the gate name string, and ``value`` is the gate time
        in nanoseconds.

        If a custom gate is specified that already exists in
        the backend properties, the ``gate_lengths`` value will override the
        gate time value from the backend properties.
        If non-default values are used gate_lengths should be a list

        Args:
            backend (Backend): backend. For BackendV2, `standard_gates` and `warnings`
                               options are ignored, and their default values are used.
            gate_error (bool): Include depolarizing gate errors (Default: True).
            readout_error (Bool): Include readout errors in model
                                  (Default: True).
            thermal_relaxation (Bool): Include thermal relaxation errors
                                       (Default: True).
            temperature (double): qubit temperature in milli-Kelvin (mK) for
                                  thermal relaxation errors (Default: 0).
            gate_lengths (list): Custom gate times for thermal relaxation errors.
                                  Used to extend or override the gate times in
                                  the backend properties (Default: None))
            gate_length_units (str): Time units for gate length values in
                                     gate_lengths. Can be 'ns', 'ms', 'us',
                                     or 's' (Default: 'ns').
            standard_gates (bool): DEPRECATED, If true return errors as standard
                                   qobj gates. If false return as unitary
                                   qobj instructions (Default: None)
            warnings (bool): PLAN TO BE DEPRECATED, Display warnings (Default: None).

        Returns:
            NoiseModel: An approximate noise model for the device backend.

        Raises:
            NoiseError: If the input backend is not valid.
        """
        if standard_gates is not None:
            warn(
                '"standard_gates" option has been deprecated as of qiskit-aer 0.10.0'
                ' and will be removed no earlier than 3 months from that release date.',
                DeprecationWarning, stacklevel=2)

        if warnings is not None:
            warn(
                '"warnings" argument will be deprecated as part of the qiskit-aer 0.12.0 and '
                'subsequently removed',
                PendingDeprecationWarning, stacklevel=2)
        else:
            warnings = True

        backend_interface_version = getattr(backend, "version", None)
        if not isinstance(backend_interface_version, int):
            backend_interface_version = 0

        target = None
        if isinstance(backend, BackendProperties):
            warn(
                'Passing BackendProperties instead of a "backend" object '
                'has been deprecated as of qiskit-aer 0.10.0 and will be '
                'removed no earlier than 3 months from that release date. '
                'Duration dependent delay relaxation noise requires a '
                'backend object.', DeprecationWarning, stacklevel=2)
            properties = backend
            basis_gates = set()
            for prop in properties.gates:
                basis_gates.add(prop.gate)
            basis_gates = list(basis_gates)
            num_qubits = len(properties.qubits)
            all_qubit_properties = [QubitProperties(t1=properties.t1(q),
                                                    t2=properties.t2(q),
                                                    frequency=properties.frequency(q))
                                    for q in range(num_qubits)]
            dt = 0  # disable delay noise if dt is unknown
        elif backend_interface_version == 2:
            if standard_gates is not None or not warnings:
                warn("When a BackendV2 is supplied, `standard_gates` and `warnings`"
                     " are ignored, and their default values are used.", UserWarning)
            properties = None
            basis_gates = backend.operation_names
            target = backend.target
            if gate_lengths:
                # Update target based on gate_lengths and gate_length_units
                target = copy.deepcopy(target)
                for op_name, qubits, value in gate_lengths:
                    prop = target[op_name][qubits]
                    prop.duration = apply_prefix(value, gate_length_units)  # convert to seconds
                    target.update_instruction_properties(op_name, qubits, prop)
            all_qubit_properties = backend.target.qubit_properties
            if not all_qubit_properties:
                warn(f"Qiskit backend {backend} has no QubitProperties, so the resulting"
                     " noise model will not include any thermal relaxation errors.", UserWarning)
            dt = backend.dt
        elif backend_interface_version <= 1:
            properties = backend.properties()
            configuration = backend.configuration()
            basis_gates = configuration.basis_gates
            all_qubit_properties = [QubitProperties(t1=properties.t1(q),
                                                    t2=properties.t2(q),
                                                    frequency=properties.frequency(q))
                                    for q in range(configuration.num_qubits)]
            dt = getattr(configuration, "dt", 0)
            if not properties:
                raise NoiseError('Qiskit backend {} does not have a '
                                 'BackendProperties'.format(backend))
        else:
            raise NoiseError('{} is not a Qiskit backend or'
                             ' BackendProperties'.format(backend))

        noise_model = NoiseModel(basis_gates=basis_gates)

        # Add single-qubit readout errors
        if readout_error:
            for qubits, error in basic_device_readout_errors(properties, target=target):
                noise_model.add_readout_error(error, qubits, warnings=warnings)

        # Add gate errors
        with catch_warnings():
            filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module="qiskit_aer.noise"
            )
            filterwarnings(
                "ignore",
                category=PendingDeprecationWarning,
                module="qiskit_aer.noise"
            )
            gate_errors = basic_device_gate_errors(
                properties,
                gate_error=gate_error,
                thermal_relaxation=thermal_relaxation,
                gate_lengths=gate_lengths,
                gate_length_units=gate_length_units,
                temperature=temperature,
                standard_gates=standard_gates,
                warnings=warnings,
                target=target,
            )
        for name, qubits, error in gate_errors:
            noise_model.add_quantum_error(error, name, qubits, warnings=warnings)

        if thermal_relaxation and all_qubit_properties:
            # Add delay errors via RelaxationNiose pass
            try:
                excited_state_populations = [
                    _excited_population(
                        freq=q.frequency, temperature=temperature
                    ) for q in all_qubit_properties]
            except BackendPropertyError:
                excited_state_populations = None
            try:
                t1s = [prop.t1 for prop in all_qubit_properties]
                t2s = [_truncate_t2_value(prop.t1, prop.t2) for prop in all_qubit_properties]
                delay_pass = RelaxationNoisePass(
                    t1s=t1s,
                    t2s=t2s,
                    dt=dt,
                    op_types=Delay,
                    excited_state_populations=excited_state_populations
                )
                noise_model._custom_noise_passes.append(delay_pass)
            except BackendPropertyError:
                # Device does not have the required T1 or T2 information
                # in its properties
                pass

        return noise_model

    def is_ideal(self):  # pylint: disable=too-many-return-statements
        """Return True if the noise model has no noise terms."""
        # Get default errors
        if self._default_quantum_errors:
            return False
        if self._default_readout_error:
            return False
        if self._local_quantum_errors:
            return False
        if self._local_readout_errors:
            return False
        if self._nonlocal_quantum_errors:
            return False
        if self._custom_noise_passes:
            return False
        return True

    def __repr__(self):
        """Noise model repr"""
        return "<NoiseModel on {}>".format(list(self._noise_instructions))

    def __str__(self):
        """Noise model string representation"""

        # Check if noise model is ideal
        if self.is_ideal():
            return "NoiseModel: Ideal"

        # Get default errors
        default_error_ops = []
        for inst in self._default_quantum_errors:
            default_error_ops.append('{}'.format(inst))
        if self._default_readout_error is not None:
            if 'measure' not in default_error_ops:
                default_error_ops.append('measure')

        # Get local errors
        local_error_ops = []
        for inst, dic in self._local_quantum_errors.items():
            for qubits in dic.keys():
                local_error_ops.append((inst, qubits))
        for qubits in self._local_readout_errors:
            tmp = ('measure', qubits)
            if tmp not in local_error_ops:
                local_error_ops.append(tmp)

        # Get nonlocal errors
        nonlocal_error_ops = []
        for inst, dic in self._nonlocal_quantum_errors.items():
            for qubits, errors in dic.items():
                for noise_qubits in errors:
                    nonlocal_error_ops.append((inst, qubits, noise_qubits))

        output = "NoiseModel:"
        output += "\n  Basis gates: {}".format(self.basis_gates)
        if self._noise_instructions:
            output += "\n  Instructions with noise: {}".format(
                list(self._noise_instructions))
        if self._noise_qubits:
            output += "\n  Qubits with noise: {}".format(
                list(self._noise_qubits))
        if default_error_ops:
            output += "\n  All-qubits errors: {}".format(default_error_ops)
        if local_error_ops:
            output += "\n  Specific qubit errors: {}".format(
                local_error_ops)
        if nonlocal_error_ops:
            output += "\n  Non-local specific qubit errors: {}".format(
                nonlocal_error_ops)
        return output

    def __eq__(self, other):
        """Test if two noise models are equal."""
        # This returns True if both noise models have:
        # the same basis_gates
        # the same noise_qubits
        # the same noise_instructions
        if (not isinstance(other, NoiseModel) or
                self.basis_gates != other.basis_gates or
                self.noise_qubits != other.noise_qubits or
                self.noise_instructions != other.noise_instructions):
            return False
        # Check default readout errors is equal
        if not self._readout_errors_equal(other):
            return False
        # Check quantum errors equal
        if not self._all_qubit_quantum_errors_equal(other):
            return False
        if not self._local_quantum_errors_equal(other):
            return False
        if not self._nonlocal_quantum_errors_equal(other):
            return False
        # If we made it here they are equal
        return True

    def reset(self):
        """Reset the noise model."""
        self.__init__()

    def add_basis_gates(self, instructions, warnings=False):
        """Add additional gates to the noise model basis_gates.

        This should be used to add any gates that are identified by a
        custom gate label in the noise model.

        Args:
            instructions (list[str] or
                          list[Instruction]): the instructions error applies to.
            warnings (bool): [DEPRECATED] display warning if instruction is not in
                             QasmSimulator basis_gates (Default: False).
        """
        for name, _ in self._instruction_names_labels(instructions):
            # If the instruction is in the default basis gates for the
            # AerSimulator we add it to the basis gates.
            if name in BASIS_GATES['automatic']:
                if name not in ['measure', 'reset', 'initialize',
                                'kraus', 'superop', 'roerror']:
                    self._basis_gates.add(name)
            elif warnings:
                warn('"warnings" option has been deprecated as of qiskit-aer 0.10.0'
                     ' and will be removed no earlier than 3 months from that release date.',
                     DeprecationWarning, stacklevel=2)
                logger.warning(
                    "Warning: Adding a gate \"%s\" to basis_gates which is "
                    "not in AerSimulator basis_gates.", name)

    def add_all_qubit_quantum_error(self, error, instructions, warnings=True):
        """
        Add a quantum error to the noise model that applies to all qubits.

        Args:
            error (QuantumError): the quantum error object.
            instructions (str or list[str] or
                          Instruction or
                          list[Instruction]): the instructions error applies to.
            warnings (bool): Display warning if appending to an instruction that
                             already has an error (Default: True).

        Raises:
            NoiseError: if the input parameters are invalid.

        Additional Information:
            If the error object is ideal it will not be added to the model.
        """
        # Format input as QuantumError
        if not isinstance(error, QuantumError):
            try:
                error = QuantumError(error)
            except NoiseError as ex:
                raise NoiseError("Input is not a valid quantum error.") from ex
        # Check if error is ideal and if so don't add to the noise model
        if error.ideal():
            return

        # Add instructions
        for name, label in self._instruction_names_labels(instructions):
            self._check_number_of_qubits(error, name)
            if label in self._default_quantum_errors:
                new_error = self._default_quantum_errors[label].compose(error)
                self._default_quantum_errors[label] = new_error
                if warnings:
                    logger.warning(
                        "WARNING: all-qubit error already exists for "
                        "instruction \"%s\", "
                        "composing with additional error.", label)
            else:
                self._default_quantum_errors[label] = error
            # Check if a specific qubit error has been applied for this instruction
            if label in self._local_quantum_errors:
                local_qubits = self._keys2str(
                    self._local_quantum_errors[label].keys())
                if warnings:
                    logger.warning(
                        "WARNING: all-qubit error for instruction "
                        "\"%s\" will not apply to qubits: "
                        "%s as specific error already exists.", label,
                        local_qubits)
            self._noise_instructions.add(label)
            self.add_basis_gates(name)

    def add_quantum_error(self, error, instructions, qubits, warnings=True):
        """
        Add a quantum error to the noise model.

        Args:
            error (QuantumError): the quantum error object.
            instructions (str or list[str] or
                          Instruction or
                          list[Instruction]): the instructions error applies to.
            qubits (Sequence[int]): qubits instruction error applies to.
            warnings (bool): Display warning if appending to an instruction that
                             already has an error (Default: True).

        Raises:
            NoiseError: if the input parameters are invalid.

        Additional Information:
            If the error object is ideal it will not be added to the model.
        """
        # Error checking
        if not isinstance(error, QuantumError):
            try:
                error = QuantumError(error)
            except NoiseError as ex:
                raise NoiseError("Input is not a valid quantum error.") from ex
        try:
            qubits = tuple(qubits)
        except TypeError as ex:
            raise NoiseError("Qubits must be convertible to a tuple of integers") from ex
        # Check if error is ideal and if so don't add to the noise model
        if error.ideal():
            return
        # Add noise qubits
        for qubit in qubits:
            self._noise_qubits.add(qubit)
        # Add instructions
        for name, label in self._instruction_names_labels(instructions):
            self._check_number_of_qubits(error, name)
            if not isinstance(label, str):
                raise NoiseError("Qobj invalid instructions.")
            # Check number of qubits is correct for standard instructions
            self._check_number_of_qubits(error, name)
            if label in self._local_quantum_errors:
                qubit_dict = self._local_quantum_errors[label]
            else:
                qubit_dict = {}

            # Convert qubits list to hashable string
            if error.num_qubits != len(qubits):
                raise NoiseError("Number of qubits ({}) does not match "
                                 " the error size ({})".format(
                                     len(qubits), error.num_qubits))
            if qubits in qubit_dict:
                new_error = qubit_dict[qubits].compose(error)
                qubit_dict[qubits] = new_error
                if warnings:
                    logger.warning(
                        "WARNING: quantum error already exists for "
                        "instruction \"%s\" on qubits %s "
                        ", appending additional error.", label, qubits)
            else:
                qubit_dict[qubits] = error
            # Add updated dictionary
            self._local_quantum_errors[label] = qubit_dict

            # Check if all-qubit error is already defined for this instruction
            if label in self._default_quantum_errors:
                if warnings:
                    logger.warning(
                        "WARNING: Specific error for instruction \"%s\" "
                        "on qubits %s overrides previously defined "
                        "all-qubit error for these qubits.", label, qubits)
            self._noise_instructions.add(label)
            self.add_basis_gates(name)

    def add_nonlocal_quantum_error(self,
                                   error,
                                   instructions,
                                   qubits,
                                   noise_qubits,
                                   warnings=True):
        """
        Add a non-local quantum error to the noise model (DEPRECATED).

        .. deprecated:: 0.9.0

            Adding nonlocal noise to a noise model is deprecated and will
            be removed no earlier than 3 months from the qiskit-aer 0.9.0
            release date. To add non-local noise to a circuit you should
            write a custom qiskit transpiler pass.

        Args:
            error (QuantumError): the quantum error object.
            instructions (str or list[str] or
                          Instruction or
                          list[Instruction]): the instructions error applies to.
            qubits (Sequence[int]): qubits instruction error applies to.
            noise_qubits (Sequence[int]): Specify the exact qubits the error
                                      should be applied to if different
                                      to the instruction qubits.
            warnings (bool): Display warning if appending to an instruction that
                             already has an error (Default: True).

        Raises:
            NoiseError: if the input parameters are invalid.

        Additional Information:
            If the error object is ideal it will not be added to the model.
        """
        warn('Adding nonlocal noise to a noise model is deprecated as of'
             ' qiskit-aer 0.9.0 and will be removed no earlier than 3'
             ' months from that release date. To add non-local noise to'
             ' a circuit you should write a custom qiskit transpiler pass.',
             DeprecationWarning)

        # Error checking
        if not isinstance(error, QuantumError):
            try:
                error = QuantumError(error)
            except NoiseError as ex:
                raise NoiseError("Input is not a valid quantum error.") from ex
        try:
            qubits = tuple(qubits)
            noise_qubits = tuple(noise_qubits)
        except TypeError as ex:
            raise NoiseError("Qubits must be convertible to a tuple of integers") from ex
        # Check if error is ideal and if so don't add to the noise model
        if error.ideal():
            return
        # Add noise qubits
        for qubit in qubits:
            self._noise_qubits.add(qubit)
        for qubit in noise_qubits:
            self._noise_qubits.add(qubit)
        # Add instructions
        for name, label in self._instruction_names_labels(instructions):
            if label in self._nonlocal_quantum_errors:
                gate_qubit_dict = self._nonlocal_quantum_errors[label]
            else:
                gate_qubit_dict = {}
            if qubits in gate_qubit_dict:
                noise_qubit_dict = gate_qubit_dict[qubits]
                if noise_qubits in noise_qubit_dict:
                    new_error = noise_qubit_dict[noise_qubits].compose(error)
                    noise_qubit_dict[noise_qubits] = new_error
                else:
                    noise_qubit_dict[noise_qubits] = error
                gate_qubit_dict[qubits] = noise_qubit_dict
                if warnings:
                    logger.warning(
                        "Warning: nonlocal error already exists for "
                        "instruction \"%s\" on qubits %s."
                        "Composing additional error.", label, qubits)
            else:
                gate_qubit_dict[qubits] = {noise_qubits: error}
            # Add updated dictionary
            self._nonlocal_quantum_errors[label] = gate_qubit_dict
            self._noise_instructions.add(label)
            self.add_basis_gates(name, warnings=False)

    def add_all_qubit_readout_error(self, error, warnings=True):
        """
        Add a single-qubit readout error that applies measure on all qubits.

        Args:
            error (ReadoutError): the quantum error object.
            warnings (bool): Display warning if appending to an instruction that
                             already has an error (Default: True)

        Raises:
            NoiseError: if the input parameters are invalid.

        Additional Information:
            If the error object is ideal it will not be added to the model.
        """

        # Error checking
        if not isinstance(error, ReadoutError):
            try:
                error = ReadoutError(error)
            except NoiseError as ex:
                raise NoiseError("Input is not a valid readout error.") from ex

        # Check if error is ideal and if so don't add to the noise model
        if error.ideal():
            return

        # Check number of qubits is correct for standard instructions
        if error.number_of_qubits != 1:
            raise NoiseError(
                "All-qubit readout errors must defined as single-qubit errors."
            )
        if self._default_readout_error is not None:
            if warnings:
                logger.warning(
                    "WARNING: all-qubit readout error already exists, "
                    "overriding with new readout error.")
        self._default_readout_error = error

        # Check if a specific qubit error has been applied for this instruction
        if self._local_readout_errors:
            local_qubits = self._keys2str(self._local_readout_errors.keys())
            if warnings:
                logger.warning(
                    "WARNING: The all-qubit readout error will not "
                    "apply to measure of qubits qubits: %s "
                    "as specific readout errors already exist.", local_qubits)
        self._noise_instructions.add("measure")

    def add_readout_error(self, error, qubits, warnings=True):
        """
        Add a readout error to the noise model.

        Args:
            error (ReadoutError): the quantum error object.
            qubits (list[int] or tuple[int]): qubits instruction error applies to.
            warnings (bool): Display warning if appending to an instruction that
                             already has an error [Default: True]

        Raises:
            NoiseError: if the input parameters are invalid.

        Additional Information:
            If the error object is ideal it will not be added to the model.
        """

        # Error checking
        if not isinstance(error, ReadoutError):
            try:
                error = ReadoutError(error)
            except NoiseError as ex:
                raise NoiseError("Input is not a valid readout error.") from ex
        try:
            qubits = tuple(qubits)
        except TypeError as ex:
            raise NoiseError("Qubits must be convertible to a tuple of integers") from ex

        # Check if error is ideal and if so don't add to the noise model
        if error.ideal():
            return

        # Add noise qubits
        for qubit in qubits:
            self._noise_qubits.add(qubit)

        # Check error matches qubit size
        if error.number_of_qubits != len(qubits):
            raise NoiseError(
                "Number of qubits ({}) does not match the readout "
                "error size ({})".format(len(qubits), error.number_of_qubits))
        # Check if we are overriding a previous error
        if qubits in self._local_readout_errors:
            if warnings:
                logger.warning(
                    "WARNING: readout error already exists for qubits "
                    "%s, overriding with new readout error.", qubits)
        self._local_readout_errors[qubits] = error

        # Check if all-qubit readout error is already defined
        if self._default_readout_error is not None:
            if warnings:
                logger.warning(
                    "WARNING: Specific readout error on qubits "
                    "%s overrides previously defined "
                    "all-qubit readout error for these qubits.", qubits)
        self._noise_instructions.add("measure")

    def to_dict(self, serializable=False):
        """
        Return the noise model as a dictionary.

        Args:
            serializable (bool): if `True`, return a dict containing only types
                that can be serializable by the stdlib `json` module.

        Returns:
            dict: a dictionary for a noise model.
        """
        error_list = []

        # Add default quantum errors
        for name, error in self._default_quantum_errors.items():
            error_dict = error.to_dict()
            error_dict["operations"] = [name]
            error_list.append(error_dict)

        # Add specific qubit errors
        for name, qubit_dict in self._local_quantum_errors.items():
            for qubits, error in qubit_dict.items():
                error_dict = error.to_dict()
                error_dict["operations"] = [name]
                error_dict["gate_qubits"] = [qubits]
                error_list.append(error_dict)

        # Add non-local errors
        for name, qubit_dict in self._nonlocal_quantum_errors.items():
            for qubits, noise_qubit_dict in qubit_dict.items():
                for noise_qubits, error in noise_qubit_dict.items():
                    error_dict = error.to_dict()
                    error_dict["operations"] = [name]
                    error_dict["gate_qubits"] = [qubits]
                    error_dict["noise_qubits"] = [noise_qubits]
                    error_list.append(error_dict)

        # Add default readout error
        if self._default_readout_error is not None:
            error_dict = self._default_readout_error.to_dict()
            error_list.append(error_dict)

        # Add local readout error
        for qubits, error in self._local_readout_errors.items():
            error_dict = error.to_dict()
            error_dict["gate_qubits"] = [qubits]
            error_list.append(error_dict)

        ret = {"errors": error_list}
        if serializable:
            ret = json.loads(json.dumps(ret, cls=AerJSONEncoder))

        return ret

    @staticmethod
    def from_dict(noise_dict):
        """
        Load NoiseModel from a dictionary.

        Args:
            noise_dict (dict): A serialized noise model.

        Returns:
            NoiseModel: the noise model.

        Raises:
            NoiseError: if dict cannot be converted to NoiseModel.
        """
        warn('from_dict has been deprecated as of qiskit-aer 0.10.0'
             ' and will be removed no earlier than 3 months from that release date.',
             DeprecationWarning, stacklevel=2)
        # Return noise model
        noise_model = NoiseModel()

        # Get error terms
        errors = noise_dict.get('errors', [])

        for error in errors:
            error_type = error['type']

            # Add QuantumError
            if error_type == 'qerror':
                noise_ops = tuple(
                    zip(error['instructions'], error['probabilities']))
                instruction_names = error['operations']
                all_gate_qubits = error.get('gate_qubits', None)
                all_noise_qubits = error.get('noise_qubits', None)
                with catch_warnings():
                    filterwarnings("ignore",
                                   category=DeprecationWarning,
                                   module="qiskit_aer.noise")
                    qerror = QuantumError(noise_ops)
                qerror._id = error.get('id', None) or qerror.id
                if all_gate_qubits is not None:
                    for gate_qubits in all_gate_qubits:
                        # Load non-local quantum error
                        if all_noise_qubits is not None:
                            for noise_qubits in all_noise_qubits:
                                noise_model.add_nonlocal_quantum_error(
                                    qerror,
                                    instruction_names,
                                    gate_qubits,
                                    noise_qubits,
                                    warnings=False)
                        # Add local quantum error
                        else:
                            noise_model.add_quantum_error(
                                qerror,
                                instruction_names,
                                gate_qubits,
                                warnings=False)
                else:
                    # Add all-qubit quantum error
                    noise_model.add_all_qubit_quantum_error(
                        qerror, instruction_names, warnings=False)

            # Add ReadoutError
            elif error_type == 'roerror':
                probabilities = error['probabilities']
                all_gate_qubits = error.get('gate_qubits', None)
                roerror = ReadoutError(probabilities)
                # Add local readout error
                if all_gate_qubits is not None:
                    for gate_qubits in all_gate_qubits:
                        noise_model.add_readout_error(
                            roerror, gate_qubits, warnings=False)
                # Add all-qubit readout error
                else:
                    noise_model.add_all_qubit_readout_error(
                        roerror, warnings=False)
            # Invalid error type
            else:
                raise NoiseError("Invalid error type: {}".format(error_type))
        return noise_model

    def _instruction_names_labels(self, instructions):
        """Return two lists of instruction name strings and label strings."""
        if not isinstance(instructions, (list, tuple)):
            instructions = [instructions]
        names_labels = []
        for inst in instructions:
            # If instruction does not have a label we use the name
            # as the label
            if isinstance(inst, Instruction):
                name = inst.name
                label = getattr(inst, 'label', inst.name)
                names_labels.append((name, label))
            elif isinstance(inst, str):
                names_labels.append((inst, inst))
            else:
                raise NoiseError('Invalid instruction type {}'.format(inst))
        return names_labels

    def _check_number_of_qubits(self, error, name):
        """
        Check if error is corrected number of qubits for standard instruction.

        Args:
            error (QuantumError): the quantum error object.
            name (str): qobj instruction name to apply error to.

        Raises:
            NoiseError: If instruction and error qubit number do not match.
        """

        def error_message(gate_qubits):
            msg = "{} qubit QuantumError".format(error.num_qubits) + \
                  " cannot be applied to {} qubit".format(gate_qubits) + \
                  " instruction \"{}\".".format(name)
            return msg

        if name in self._1qubit_instructions and error.num_qubits != 1:
            raise NoiseError(error_message(1))
        if name in self._2qubit_instructions and error.num_qubits != 2:
            raise NoiseError(error_message(2))
        if name in self._3qubit_instructions and error.num_qubits != 3:
            raise NoiseError(error_message(3))

    def _keys2str(self, keys):
        """Convert dicitonary keys to comma seperated print string."""
        tmp = "".join(["{}, ".format(key) for key in keys])
        return tmp[:-2]

    def _readout_errors_equal(self, other):
        """Check two noise models have equal readout errors"""
        # Check default readout error is equal
        if self._default_readout_error != other._default_readout_error:
            return False
        # Check local readout errors are equal
        if sorted(self._local_readout_errors.keys()) != sorted(
                other._local_readout_errors.keys()):
            return False
        for key in self._local_readout_errors:
            if self._local_readout_errors[key] != other._local_readout_errors[
                    key]:
                return False
        return True

    def _all_qubit_quantum_errors_equal(self, other):
        """Check two noise models have equal local quantum errors"""
        if sorted(self._default_quantum_errors.keys()) != sorted(
                other._default_quantum_errors.keys()):
            return False
        for key in self._default_quantum_errors:
            if self._default_quantum_errors[
                    key] != other._default_quantum_errors[key]:
                return False
        return True

    def _local_quantum_errors_equal(self, other):
        """Check two noise models have equal local quantum errors"""
        if sorted(self._local_quantum_errors.keys()) != sorted(
                other._local_quantum_errors.keys()):
            return False
        for key in self._local_quantum_errors:
            inner_dict1 = self._local_quantum_errors[key]
            inner_dict2 = other._local_quantum_errors[key]
            if sorted(inner_dict1.keys()) != sorted(inner_dict2.keys()):
                return False
            for inner_key in inner_dict1:
                if inner_dict1[inner_key] != inner_dict2[inner_key]:
                    return False
            if self._local_quantum_errors[key] != other._local_quantum_errors[
                    key]:
                return False
        return True

    def _nonlocal_quantum_errors_equal(self, other):
        """Check two noise models have equal non-local quantum errors"""
        if sorted(self._nonlocal_quantum_errors.keys()) != sorted(
                other._nonlocal_quantum_errors.keys()):
            return False
        for key in self._nonlocal_quantum_errors:
            inner_dict1 = self._nonlocal_quantum_errors[key]
            inner_dict2 = other._nonlocal_quantum_errors[key]
            if sorted(inner_dict1.keys()) != sorted(inner_dict2.keys()):
                return False
            for inner_key in inner_dict1:
                iinner_dict1 = inner_dict1[inner_key]
                iinner_dict2 = inner_dict2[inner_key]
                if sorted(iinner_dict1.keys()) != sorted(iinner_dict2.keys()):
                    return False
                for iinner_key in iinner_dict1:
                    if iinner_dict1[iinner_key] != iinner_dict2[iinner_key]:
                        return False
        return True

    def _pass_manager(self) -> Optional[PassManager]:
        """
        Return the pass manager that add custom noises defined as noise passes
        (stored in the _custom_noise_passes field). Note that the pass manager
        does not include passes to add other noises (stored in the different field).
        """
        passes = []
        passes.extend(self._custom_noise_passes)
        if len(passes) > 0:
            return PassManager(passes)
        return None
