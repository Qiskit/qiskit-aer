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

import json
import logging

from qiskit.circuit import Instruction
from qiskit.providers.aer.backends.aerbackend import AerJSONEncoder
from qiskit.providers.aer.backends.qasm_simulator import QasmSimulator

from .noiseerror import NoiseError
from .errors.quantum_error import QuantumError
from .errors.readout_error import ReadoutError
from ..utils.helpers import deprecation

logger = logging.getLogger(__name__)


class NoiseModel:
    """Noise model class for Qiskit Aer simulators."""

    # Get the default basis gates for the Qiskit Aer Qasm Simulator
    # this is used to decide what are instructions for a noise model
    # and what are labels for other named instructions
    QASMSIMULATOR_BASIS_GATES = QasmSimulator.DEFAULT_CONFIGURATION[
        'basis_gates']

    # Checks for standard 1-3 qubit instructions
    _1qubit_instructions = set([
        "x90", "u1", "u2", "u3", "U", "id", "x", "y", "z", "h", "s", "sdg",
        "t", "tdg"
    ])
    _2qubit_instructions = set(["CX", "cx", "cz", "swap"])
    _3qubit_instructions = set(["ccx"])

    def __init__(self, basis_gates=None):
        """Initialize an empty noise model.

        Args:
            basis_gates (list[str] or None): Specify an initial basis_gates
                for the noise model. If None a default value of ['id', 'u3', 'cx']
                is used (Default: None).

        Additional Information
        ----------------------
        Errors added to the noise model will have thier instruction appended to
        the noise model basis_gates if the instruction is in the QasmSimulator
        basis_gates. If the instruction is not in the QasmSimulator basis_gates
        it is assumbed to be a label for a standard gate, and that gate should
        be added to the NoiseModel basis_gates either using the init method, or
        the `add_basis_Gates` method.
        """
        if basis_gates is None:
            # Default basis gates is id, u3, cx so that all standard
            # non-identity instructions can be unrolled to u3, cx,
            # and identities won't be unrolled to u3.
            self._basis_gates = set(['id', 'u3', 'cx'])
        else:
            self._basis_gates = set(self._instruction_names(basis_gates))
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
        # dict(str: dict(str: QuantumError))
        # where the outer keys are the instruction str label and the
        # inner dict keys are the gate qubits
        self._local_quantum_errors = {}
        # Non-local quantum errors are stored as:
        # dict(str: dict(str: dict(str: QuantumError)))
        # where the outer keys are the instruction str label, the middle dict
        # keys are the gate qubits, and the inner most dict keys are
        # the noise qubits.
        self._nonlocal_quantum_errors = {}
        # Default (all-qubit) readout error is stored as a single
        # ReadoutError object since there may only be one defined.
        self._default_readout_error = None
        # Local readout errors are stored as:
        # dict(str: ReadoutError)
        # where the dict keys are the gate qubits.
        self._local_readout_errors = {}
        self._x90_gates = []

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

    def __repr__(self):
        """Display noise model"""

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
            for q_str in dic.keys():
                local_error_ops.append((inst, self._str2qubits(q_str)))
        for q_str in self._local_readout_errors:
            tmp = ('measure', self._str2qubits(q_str))
            if tmp not in local_error_ops:
                local_error_ops.append(tmp)

        # Get nonlocal errors
        nonlocal_error_ops = []
        for inst, dic in self._nonlocal_quantum_errors.items():
            for q_str, errors in dic.items():
                for nq_str in errors:
                    nonlocal_error_ops.append((inst, self._str2qubits(q_str),
                                               self._str2qubits(nq_str)))

        output = "NoiseModel:"
        if default_error_ops == [] and local_error_ops == [] and nonlocal_error_ops == []:
            output += " Ideal"
        else:
            output += "\n  Basis gates: {}".format(self.basis_gates)
            if self._noise_instructions:
                output += "\n  Instructions with noise: {}".format(
                    list(self._noise_instructions))
            if self._noise_qubits:
                output += "\n  Qubits with noise: {}".format(
                    list(self._noise_qubits))
            if self._x90_gates:
                output += "\n  X-90 based single qubit gates: {}".format(
                    list(self._x90_gates))
            if default_error_ops != []:
                output += "\n  All-qubits errors: {}".format(default_error_ops)
            if local_error_ops != []:
                output += "\n  Specific qubit errors: {}".format(
                    local_error_ops)
            if nonlocal_error_ops != []:
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

    def add_basis_gates(self, instructions, warnings=True):
        """Add additional gates to the noise model basis_gates.

        This should be used to add any gates that are identified by a
        custom gate label in the noise model.

        Args:
            instructions (list[str] or
                          list[Instruction]): the instructions error applies to.
            warnings (bool): display warning if instruction is not in
                             QasmSimulator basis_gates (Default: True).
        """
        names = self._instruction_names(instructions)
        for inst in names:
            # If the instruction is in the default basis gates for the
            # QasmSimulator we add it to the basis gates.
            if inst in self.QASMSIMULATOR_BASIS_GATES:
                if inst not in ['measure', 'reset']:
                    self._basis_gates.add(inst)
            elif warnings:
                logger.warning(
                    "Warning: Adding a gate \"%s\" to basis_gates which is "
                    "not in QasmSimulator basis_gates.", inst)

    def set_x90_single_qubit_gates(self, instructions):
        """
        Declares X90 based gates for noise model.

        Args:
            instructions (list[str] or
                          list[Instruction]): the instructions error applies to.

        Raises:
            NoiseError: if the input instructions are not valid.
        """
        names = self._instruction_names(instructions)
        for inst in names:
            # Add X-90 based gate to noisy gates
            self._noise_instructions.add(inst)
            self._basis_gates.add(inst)
        self._x90_gates = instructions

    def add_all_qubit_quantum_error(self, error, instructions, warnings=True):
        """
        Add a quantum error to the noise model that applies to all qubits.

        Args:
            error (QuantumError): the quantum error object.
            instructions (str or list[str] or
                          Instruction or
                          list[Instruction]): the instructions error applies to.
            warnings (bool): Display warning if appending to an instruciton that
                             already has an error (Default: True).

        Raises:
            NoiseError: if the input parameters are invalid.

        Additional Information
        ----------------------
        If the error object is ideal it will not be added to the model.
        """
        instruction_names = self._instruction_names(instructions)
        # Format input as QuantumError
        if not isinstance(error, QuantumError):
            try:
                error = QuantumError(error)
            except NoiseError:
                raise NoiseError("Input is not a valid quantum error.")
        # Check if error is ideal and if so don't add to the noise model
        if error.ideal():
            return

        # Add instructions
        for inst in instruction_names:
            self._check_number_of_qubits(error, inst)
            if inst in self._default_quantum_errors:
                new_error = self._default_quantum_errors[inst].compose(error)
                self._default_quantum_errors[inst] = new_error
                if warnings:
                    logger.warning(
                        "WARNING: all-qubit error already exists for "
                        "instruction \"%s\", "
                        "composing with additional error.", inst)
            else:
                self._default_quantum_errors[inst] = error
            # Check if a specific qubit error has been applied for this instruction
            if inst in self._local_quantum_errors:
                local_qubits = self._keys2str(
                    self._local_quantum_errors[inst].keys())
                if warnings:
                    logger.warning(
                        "WARNING: all-qubit error for instruction "
                        "\"%s\" will not apply to qubits: "
                        "%s as specific error already exists.", inst,
                        local_qubits)
            self._noise_instructions.add(inst)
            self.add_basis_gates(inst, warnings=False)

    def add_quantum_error(self, error, instructions, qubits, warnings=True):
        """
        Add a quantum error to the noise model.

        Args:
            error (QuantumError): the quantum error object.
            instructions (str or list[str] or
                          Instruction or
                          list[Instruction]): the instructions error applies to.
            qubits (list[int]): qubits instruction error applies to.
            warnings (bool): Display warning if appending to an instruciton that
                             already has an error (Default: True).

        Raises:
            NoiseError: if the input parameters are invalid.

        Additional Information
        ----------------------
        If the error object is ideal it will not be added to the model.
        """
        instruction_names = self._instruction_names(instructions)
        if not isinstance(qubits, (list, tuple)):
            raise NoiseError("Qubits must be a list of integers.")
        # Error checking
        if not isinstance(error, QuantumError):
            try:
                error = QuantumError(error)
            except NoiseError:
                raise NoiseError("Input is not a valid quantum error.")
        # Check if error is ideal and if so don't add to the noise model
        if error.ideal():
            return
        # Add noise qubits
        for qubit in qubits:
            self._noise_qubits.add(qubit)
        # Add instructions
        for inst in instruction_names:
            if not isinstance(inst, str):
                raise NoiseError("Qobj invalid instructions.")
            # Check number of qubits is correct for standard instructions
            self._check_number_of_qubits(error, inst)
            if inst in self._local_quantum_errors:
                qubit_dict = self._local_quantum_errors[inst]
            else:
                qubit_dict = {}

            # Convert qubits list to hashable string
            qubits_str = self._qubits2str(qubits)
            if error.number_of_qubits != len(qubits):
                raise NoiseError("Number of qubits ({}) does not match "
                                 " the error size ({})".format(
                                     len(qubits), error.number_of_qubits))
            if qubits_str in qubit_dict:
                new_error = qubit_dict[qubits_str].compose(error)
                qubit_dict[qubits_str] = new_error
                if warnings:
                    logger.warning(
                        "WARNING: quantum error already exists for "
                        "instruction \"%s\" on qubits %s "
                        ", appending additional error.", inst, qubits)
            else:
                qubit_dict[qubits_str] = error
            # Add updated dictionary
            self._local_quantum_errors[inst] = qubit_dict

            # Check if all-qubit error is already defined for this instruction
            if inst in self._default_quantum_errors:
                if warnings:
                    logger.warning(
                        "WARNING: Specific error for instruction \"%s\" "
                        "on qubits %s overrides previously defined "
                        "all-qubit error for these qubits.", inst, qubits)
            self._noise_instructions.add(inst)
            self.add_basis_gates(inst, warnings=False)

    def add_nonlocal_quantum_error(self,
                                   error,
                                   instructions,
                                   qubits,
                                   noise_qubits,
                                   warnings=True):
        """
        Add a non-local quantum error to the noise model.

        Args:
            error (QuantumError): the quantum error object.
            instructions (str or list[str] or
                          Instruction or
                          list[Instruction]): the instructions error applies to.
            qubits (list[int]): qubits instruction error applies to.
            noise_qubits (list[int]): Specify the exact qubits the error
                                      should be applied to if different
                                      to the instruction qubits.
            warnings (bool): Display warning if appending to an instruciton that
                             already has an error (Default: True).

        Raises:
            NoiseError: if the input parameters are invalid.

        Additional Information:
            If the error object is ideal it will not be added to the model.
        """
        instruction_names = self._instruction_names(instructions)
        if not isinstance(noise_qubits, (list, tuple)):
            raise NoiseError("Noise qubits must be a list of integers.")
        # Error checking
        if not isinstance(error, QuantumError):
            try:
                error = QuantumError(error)
            except NoiseError:
                raise NoiseError("Input is not a valid quantum error.")
        if not isinstance(qubits, (list, tuple)):
            raise NoiseError("Qubits must be a list of integers.")
        # Check if error is ideal and if so don't add to the noise model
        if error.ideal():
            return
        # Add noise qubits
        for qubit in qubits:
            self._noise_qubits.add(qubit)
        for qubit in noise_qubits:
            self._noise_qubits.add(qubit)
        # Add instructions
        for inst in instruction_names:
            if inst in self._nonlocal_quantum_errors:
                gate_qubit_dict = self._nonlocal_quantum_errors[inst]
            else:
                gate_qubit_dict = {}
            qs_str = self._qubits2str(qubits)
            nqs_str = self._qubits2str(noise_qubits)
            if qs_str in gate_qubit_dict:
                noise_qubit_dict = gate_qubit_dict[qs_str]
                if nqs_str in noise_qubit_dict:
                    new_error = noise_qubit_dict[nqs_str].compose(error)
                    noise_qubit_dict[nqs_str] = new_error
                else:
                    noise_qubit_dict[nqs_str] = error
                gate_qubit_dict[qs_str] = noise_qubit_dict
                if warnings:
                    logger.warning(
                        "Warning: nonlocal error already exists for "
                        "instruction \"%s\" on qubits %s."
                        "Composing additional error.", inst, qubits)
            else:
                gate_qubit_dict[qs_str] = {nqs_str: error}
            # Add updated dictionary
            self._nonlocal_quantum_errors[inst] = gate_qubit_dict
            self._noise_instructions.add(inst)
            self.add_basis_gates(inst, warnings=False)

    def add_all_qubit_readout_error(self, error, warnings=True):
        """
        Add a single-qubit readout error that applies measure on all qubits.

        Args:
            error (ReadoutError): the quantum error object.
            warnings (bool): Display warning if appending to an instruciton that
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
            except NoiseError:
                raise NoiseError("Input is not a valid readout error.")

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
            qubits (list[int]): qubits instruction error applies to.
            warnings (bool): Display warning if appending to an instruciton that
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
            except NoiseError:
                raise NoiseError("Input is not a valid readout error.")
        if not isinstance(qubits, (list, tuple)):
            raise NoiseError("Qubits must be a list of integers.")

        # Check if error is ideal and if so don't add to the noise model
        if error.ideal():
            return

        # Add noise qubits
        for qubit in qubits:
            self._noise_qubits.add(qubit)

        # Convert qubits list to hashable string
        qubits_str = self._qubits2str(qubits)
        # Check error matches qubit size
        if error.number_of_qubits != len(qubits):
            raise NoiseError(
                "Number of qubits ({}) does not match the readout "
                "error size ({})".format(len(qubits), error.number_of_qubits))
        # Check if we are overriding a previous error
        if qubits_str in self._local_readout_errors:
            if warnings:
                logger.warning(
                    "WARNING: readout error already exists for qubits "
                    "%s, overriding with new readout error.", qubits)
        self._local_readout_errors[qubits_str] = error

        # Check if all-qubit readout error is already defined
        if self._default_readout_error is not None:
            if warnings:
                logger.warning(
                    "WARNING: Specific readout error on qubits "
                    "%s overrides previously defined "
                    "all-qubit readout error for these qubits.", qubits)
        self._noise_instructions.add("measure")

    def as_dict(self, serializable=False):
        """
        DEPRECATED: Use to_dict()
        Returns a dictionary for noise model.

        Args:
            serializable (bool): if `True`, return a dict containing only types
                that can be serializable by the stdlib `json` module.

        Returns:
            dict: a dictionary for a noise model.
        """
        deprecation("NoiseModel::as_dict() method is deprecated and will be removed after 0.3."
                    "Use '.to_dict()' instead")
        self.to_dict(serializable)

    def to_dict(self, serializable=False):
        """
        Return dictionary for noise model.

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
            for qubits_str, error in qubit_dict.items():
                error_dict = error.to_dict()
                error_dict["operations"] = [name]
                error_dict["gate_qubits"] = [self._str2qubits(qubits_str)]
                error_list.append(error_dict)

        # Add non-local errors
        for name, qubit_dict in self._nonlocal_quantum_errors.items():
            for qubits_str, noise_qubit_dict in qubit_dict.items():
                for noise_qubits_str, error in noise_qubit_dict.items():
                    error_dict = error.to_dict()
                    error_dict["operations"] = [name]
                    error_dict["gate_qubits"] = [self._str2qubits(qubits_str)]
                    error_dict["noise_qubits"] = [
                        self._str2qubits(noise_qubits_str)
                    ]
                    error_list.append(error_dict)

        # Add default readout error
        if self._default_readout_error is not None:
            error_dict = self._default_readout_error.to_dict()
            error_list.append(error_dict)

        # Add local readout error
        for qubits_str, error in self._local_readout_errors.items():
            error_dict = error.to_dict()
            error_dict["gate_qubits"] = [self._str2qubits(qubits_str)]
            error_list.append(error_dict)

        ret = {"errors": error_list, "x90_gates": self._x90_gates}
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
        # Return noise model
        noise_model = NoiseModel()

        # Set X90 gates
        noise_model.set_x90_single_qubit_gates(noise_dict.get('x90_gates', []))

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
                qerror = QuantumError(noise_ops)
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

    def _instruction_names(self, instructions):
        """Return a list of instruction name strings for input instructions."""
        if not isinstance(instructions, (list, tuple)):
            instructions = [instructions]
        inst_names = []
        for inst in instructions:
            if isinstance(inst, Instruction):
                inst_names.append(inst.name)
            elif isinstance(inst, str):
                inst_names.append(inst)
            else:
                raise NoiseError('Invalid instruction type {}'.format(inst))
        return inst_names

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
            msg = "{} qubit QuantumError".format(error.number_of_qubits) + \
                  " cannot be applied to {} qubit".format(gate_qubits) + \
                  " instruction \"{}\".".format(name)
            return msg

        if name in self._1qubit_instructions and error.number_of_qubits != 1:
            raise NoiseError(error_message(1))
        if name in self._2qubit_instructions and error.number_of_qubits != 2:
            raise NoiseError(error_message(2))
        if name in self._3qubit_instructions and error.number_of_qubits != 3:
            raise NoiseError(error_message(3))

    def _qubits2str(self, qubits):
        """Convert qubits list to comma seperated qubits string."""
        return ",".join([str(q) for q in qubits])

    def _str2qubits(self, qubits_str):
        """Convert qubits string to qubits list."""
        return [int(q) for q in qubits_str.split(',')]

    def _keys2str(self, keys):
        """Convert dicitonary keys to comma seperated print string."""
        tmp = "".join(["{}, ".format(self._str2qubits(key)) for key in keys])
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
