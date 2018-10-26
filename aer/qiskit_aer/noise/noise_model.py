# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Noise model class for Qiskit Aer simulators.
"""

import logging

from .aernoiseerror import AerNoiseError
from .quantum_error import QuantumError
from .readout_error import ReadoutError

logger = logging.getLogger(__name__)


class NoiseModel:

    """
    Noise model class for Qiskit Aer simulators.
    """

    # Checks for standard 1-3 qubit operations
    _1qubit_operations = set(["x90", "u1", "u2", "u3", "U", "id", "x", "y", "z",
                              "h", "s", "sdg", "t", "tdg"])
    _2qubit_operations = set(["CX", "cx", "cz", "swap"])
    _3qubit_operations = set(["ccx"])

    def __init__(self):
        # Initialize empty quantum errors
        self._noise_instructions = set()  # Store gates with a noise model defined
        # TODO: Code would be cleaner if these were replaced with classes
        self._default_quantum_errors = {}  # Type: dict(str: list(QuantumError)
        self._local_quantum_errors = {}    # Type: dict(str: dict(str: list(QuantumError))
        self._nonlocal_quantum_errors = {}  # Type: dict(str: dict(str: list(pair(list(QuantumError), list(int)))))
        # Initialize empty readout errors
        self._default_readout_error = None    # Type: ReadoutError
        self._local_readout_errors = {}    # Type: dict(str: ReadoutError)
        self._x90_gates = []

    @property
    def noise_instructions(self):
        """Return the set of noisy instructions for this noise model."""
        return self._noise_instructions

    def set_x90_single_qubit_gates(self, operations):
        """
        Declares X90 based gates for noise model.

        Args:
            operations (list[str]): the operations error applies to.

        Raises:
            AerNoiseError: if the input operations are not valid.
        """

        if isinstance(operations, str):
            operations = [operations]
        for op in operations:
            if not isinstance(op, str):
                raise AerNoiseError("Qobj invalid operations.")
            # Add X-90 based gate to noisy gates
            self._noise_instructions.add(op)
        self._x90_gates = operations

    def add_all_qubit_quantum_error(self, error, operations):
        """
        Add a quantum error to the noise model that applies to all qubits.

        Args:
            error (QuantumError): the quantum error object.
            operations (str, list[str]): the operations error applies to.

        Raises:
            AerNoiseError: if the input parameters are invalid.
        """

        # Convert single operation to list
        if isinstance(operations, str):
            operations = (operations,)

        if not isinstance(error, QuantumError):
            raise AerNoiseError("Input is not a QuantumError object.")
        if not isinstance(operations, (list, tuple)):
            raise AerNoiseError("Qobj invalid operations.")

        # Add operations
        for op in operations:
            if not isinstance(op, str):
                raise AerNoiseError("Qobj invalid operations.")
            self._check_number_of_qubits(error, op)
            if op in self._default_quantum_errors:
                logger.warning("WARNING: all-qubit error already exists for " +
                               "operation \"{}\", ".format(op) +
                               "appending additional error.")
                self._default_quantum_errors[op].append(error)
            else:
                self._default_quantum_errors[op] = [error]
            # Check if a specific qubit error has been applied for this operation
            if op in self._local_quantum_errors:
                local_qubits = self._keys2str(self._local_quantum_errors[op].keys())
                logger.warning("WARNING: all-qubit error for operation " +
                               "\"{}\" will not apply to qubits: ".format(op) +
                               "{} as specific error already exists.".format(local_qubits))
            self._noise_instructions.add(op)

    def add_quantum_error(self, error, operations, qubits):
        """
        Add a quantum error to the noise model.

        Args:
            error (QuantumError): the quantum error object.
            operations (str, list[str]): the operations error applies to.
            qubits (list[int]): qubits operation error applies to.

        Raises:
            AerNoiseError: if the input parameters are invalid.
        """

        # Convert single operation to list
        if isinstance(operations, str):
            operations = (operations,)

        # Error checking
        if not isinstance(error, QuantumError):
            raise AerNoiseError("Input is not a QuantumError object.")
        if not isinstance(qubits, (list, tuple)):
            raise AerNoiseError("Qubits must be a list of integers.")
        if not isinstance(operations, (list, tuple)):
            raise AerNoiseError("Qobj invalid operations.")

        # Add operations
        for op in operations:
            if not isinstance(op, str):
                raise AerNoiseError("Qobj invalid operations.")
            # Check number of qubits is correct for standard operations
            self._check_number_of_qubits(error, op)
            if op in self._local_quantum_errors:
                qubit_dict = self._local_quantum_errors[op]
            else:
                qubit_dict = {}

            # Convert qubits list to hashable string
            qubits_str = self._qubits2str(qubits)
            if error.number_of_qubits != len(qubits):
                raise AerNoiseError("Number of qubits ({}) does not match".format(len(qubits)) +
                                    " the error size ({})".format(error.number_of_qubits))
            if qubits_str in qubit_dict:
                logger.warning("WARNING: quantum error already exists for " +
                               "operation \"{}\" on qubits {}".format(op, qubits) +
                               ", appending additional error.")
                qubit_dict[qubits_str].append(error)
            else:
                qubit_dict[qubits_str] = [error]
            # Add updated dictionary
            self._local_quantum_errors[op] = qubit_dict

            # Check if all-qubit error is already defined for this operation
            if op in self._default_quantum_errors:
                logger.warning("WARNING: Specific error for operation \"{}\" ".format(op) +
                               "on qubits {} overrides previously defined ".format(qubits) +
                               "all-qubit error for these qubits.")
                self._default_quantum_errors[op].append(error)
            self._noise_instructions.add(op)

    def add_nonlocal_quantum_error(self, error, operations, qubits, noise_qubits):
        """
        Add a non-local quantum error to the noise model.

        Args:
            error (QuantumError): the quantum error object.
            operations (str, list[str]): the operations error applies to.
            qubits (list[int]): qubits operation error applies to.
            noise_qubits (list[int]): Specify the exact qubits the error
                                      should be applied to if different
                                      to the operation qubits.
        Raises:
            AerNoiseError: if the input parameters are invalid.
        """

        # Convert single operation to list
        if isinstance(operations, str):
            operations = (operations,)

        # Error checking
        if not isinstance(error, QuantumError):
            raise AerNoiseError("Input is not a QuantumError object.")
        if not isinstance(qubits, (list, tuple)):
            raise AerNoiseError("Qubits must be a list of integers.")
        if not isinstance(noise_qubits, (list, tuple)):
            raise AerNoiseError("Noise qubits must be a list of integers.")
        if not isinstance(operations, (list, tuple)):
            raise AerNoiseError("Qobj invalid operations.")

        # Add operations
        for op in operations:
            if not isinstance(op, str):
                raise AerNoiseError("Qobj invalid operations.")
            # Check number of qubits is correct for standard operations
            self._check_number_of_qubits(error, op)
            if op in self._nonlocal_quantum_errors:
                qubit_dict = self._nonlocal_quantum_errors[op]
            else:
                qubit_dict = {}
            qubits_str = "".join([str(j) for j in qubits])
            if qubits_str in qubit_dict:
                logger.warning("WARNING: nonlocal error already exists for " +
                               "operation \"{}\" on qubits {}.".format(op, qubits) +
                               "Appending additional error.")
                qubit_dict[qubits_str].append((error, noise_qubits))
            else:
                qubit_dict[qubits_str] = [(error, noise_qubits)]
            # Add updated dictionary
            self._nonlocal_quantum_errors[op] = qubit_dict
            self._noise_instructions.add(op)

    def add_all_qubit_readout_error(self, error):
        """
        Add a readout error to the noise model that applies to all qubits.

        Args:
            error (ReadoutError): the quantum error object.

        Raises:
            AerNoiseError: if the input parameters are invalid.
        """

        # Error checking
        if not isinstance(error, ReadoutError):
            raise AerNoiseError("Input is not a ReadoutError object.")

        # Check number of qubits is correct for standard operations
        if error.number_of_qubits != 1:
            raise AerNoiseError("All-qubit readout errors must defined as single-qubit errors.")
        if self._default_readout_error is not None:
            logger.warning("WARNING: all-qubit readout error already exists, " +
                           "overriding with new readout error.")
        self._default_readout_error = error

        # Check if a specific qubit error has been applied for this operation
        if len(self._local_readout_errors) > 0:
            local_qubits = self._keys2str(self._local_readout_errors.keys())
            logger.warning("WARNING: The all-qubit readout error will not" +
                           "apply to measure of qubits qubits: {} ".format(local_qubits) +
                           "as specific readout errors already exist.")
        self._noise_instructions.add("measure")

    def add_readout_error(self, error, qubits):
        """
        Add a readout error to the noise model.

        Args:
            error (ReadoutError): the quantum error object.
            qubits (list[int]): qubits operation error applies to.

        Raises:
            AerNoiseError: if the input parameters are invalid.
        """

        # Error checking
        if not isinstance(error, ReadoutError):
            raise AerNoiseError("Input is not a ReadoutError object.")
        if not isinstance(qubits, (list, tuple)):
            raise AerNoiseError("Qubits must be a list of integers.")

        # Convert qubits list to hashable string
        qubits_str = self._qubits2str(qubits)
        # Check error matches qubit size
        if error.number_of_qubits != len(qubits):
            raise AerNoiseError("Number of qubits ({}) does not match".format(len(qubits)) +
                                " the readout error size ({})".format(error.number_of_qubits))
        # Check if we are overriding a previous error
        if qubits_str in self._local_readout_errors:
            logger.warning("WARNING: readout error already exists for qubits " +
                           "{}, overriding with new readout error.".format(qubits))
        self._local_readout_errors[qubits_str] = error

        # Check if all-qubit readout error is already defined
        if self._default_readout_error is not None:
            logger.warning("WARNING: Specific readout error on qubits "
                           "{} overrides previously defined ".format(qubits) +
                           "all-qubit readout error for these qubits.")
        self._noise_instructions.add("measure")

    def __repr__(self):
        """Display noise model"""
        output = "NoiseModel"

        # Get default errors
        default_error_ops = []
        for op in self._default_quantum_errors.keys():
            default_error_ops.append('{}'.format(op))
        if self._default_readout_error is not None:
            if 'measure' not in default_error_ops:
                default_error_ops.append('measure')

        # Get local errors
        local_error_ops = []
        for op, dic in self._local_quantum_errors.items():
            for q_str in dic.keys():
                local_error_ops.append((op, self._str2qubits(q_str)))
        for q_str in self._local_readout_errors.keys():
            tmp = ('meaure', self._str2qubits(q_str))
            if tmp not in local_error_ops:
                local_error_ops.append(tmp)

        # Get nonlocal errors
        nonlocal_error_ops = []
        for op, dic in self._nonlocal_quantum_errors.items():
            for q_str, errors in dic.items():
                for error in errors:
                    nonlocal_error_ops.append((op, self._str2qubits(q_str), error[1]))

        output = "Noise Model:"
        if default_error_ops == [] and local_error_ops == [] and nonlocal_error_ops == []:
            output += " Ideal"
        else:
            if len(self._noise_instructions) > 0:
                output += "\n  Instructions with noise: {}".format(list(self._noise_instructions))
            if len(self._x90_gates) > 0:
                output += "\n  X-90 based single qubit gates: {}".format(list(self._x90_gates))
            if default_error_ops != []:
                output += "\n  All-qubits errors: {}".format(default_error_ops)
            if local_error_ops != []:
                output += "\n  Specific qubit errors: {}".format(local_error_ops)
            if nonlocal_error_ops != []:
                output += "\n  Non-local specific qubit errors: {}".format(nonlocal_error_ops)
        return output

    def as_dict(self):
        """
        Return dictionary for noise model.

        Returns:
            dict: a dictionary for a noise model.
        """

        error_list = []

        # Add default quantum errors
        for operation, errors in self._default_quantum_errors.items():
            for error in errors:
                error_dict = error.as_dict()
                error_dict["operations"] = [operation]
                error_list.append(error_dict)

        # Add specific qubit errors
        for operation, qubit_dict in self._local_quantum_errors.items():
            for qubits_str, errors in qubit_dict.items():
                for error in errors:
                    error_dict = error.as_dict()
                    error_dict["operations"] = [operation]
                    error_dict["gate_qubits"] = [self._str2qubits(qubits_str)]
                    error_list.append(error_dict)

        # Add non-local errors
        for operation, qubit_dict in self._nonlocal_quantum_errors.items():
            for qubits_str, errors in qubit_dict.items():
                for error, noise_qubits in errors:
                    error_dict = error.as_dict()
                    error_dict["operations"] = [operation]
                    error_dict["gate_qubits"] = [self._str2qubits(qubits_str)]
                    error_dict["noise_qubits"] = [list(noise_qubits)]
                    error_list.append(error_dict)

        # Add default readout error
        if self._default_readout_error is not None:
            error_dict = self._default_readout_error.as_dict()
            error_list.append(error_dict)

        # Add local readout error
        for qubits_str, error in self._local_readout_errors.items():
            error_dict = error.as_dict()
            error_dict["gate_qubits"] = [self._str2qubits(qubits_str)]
            error_list.append(error_dict)

        # Convert noise instructions to basis_gates string
        basis_gates = self._noise_instructions
        basis_gates.discard("measure")  # remove measure since it is not a gate
        basis_gates = ",".join(basis_gates)

        return {"errors": error_list,
                "x90_gates": self._x90_gates,
                "basis_gates": basis_gates}

    def _check_number_of_qubits(self, error, operation):
        """
        Check if error is corrected number of qubits for standard operation.

        Args:
            error (QuantumError): the quantum error object.
            operation (str): qobj operation strings to apply error to.

        Raises:
            AerNoiseError: If operation and error qubit number do not match.
        """
        def error_message(gate_qubits):
            msg = "{} qubit QuantumError".format(error.number_of_qubits) + \
                  " cannot be applied to {} qubit".format(gate_qubits) + \
                  " operation \"{}\".".format(operation)
            return msg
        if operation in self._1qubit_operations and error.number_of_qubits != 1:
            raise AerNoiseError(error_message(1))
        if operation in self._2qubit_operations and error.number_of_qubits != 2:
            raise AerNoiseError(error_message(2))
        if operation in self._3qubit_operations and error.number_of_qubits != 3:
            raise AerNoiseError(error_message(3))

    def _qubits2str(self, qubits):
        """Convert qubits list to qubits string."""
        return "".join([str(j) for j in qubits])

    def _str2qubits(self, qubits_str):
        """Convert qubits string to qubits list."""
        return [int(s) for s in list(qubits_str)]

    def _keys2str(self, keys):
        """Convert dicitonary keys to comma seperated print string."""
        tmp = "".join(["{}, ".format(self._str2qubits(key)) for key in keys])
        return tmp[:-2]
