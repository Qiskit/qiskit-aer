# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
Quantum error class for Qiskit Aer noise model
"""
import logging
import warnings
import copy

import numpy as np

from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.kraus import Kraus
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT

from ..noiseerror import NoiseError
from .errorutils import kraus2instructions

logger = logging.getLogger(__name__)


class QuantumError:
    """
    Quantum error class for Qiskit Aer noise model

    WARNING: The init interface for this class is not finalized and may
             change in future releases. For maximum backwards compatibility
             use the QuantumError generating functions in the `noise.errors`
             module.
    """

    ATOL = ATOL_DEFAULT
    RTOL = RTOL_DEFAULT
    MAX_TOL = 1e-4

    def __init__(self,
                 noise_ops,
                 number_of_qubits=None,
                 standard_gates=True,
                 atol=ATOL_DEFAULT):
        """
        Create a quantum error for a noise model.

        Args:
            noise_ops (list): A list of noise ops. See additional information.
            number_of_qubits (int): specify the number of qubits for the
                                    error. If None this will be determined
                                    automatically (default None).
            standard_gates (bool): Check if input matrices are standard gates.
            atol (double): Threshold for testing if probabilities are
                           equal to 0 or 1 (Default: 1e-8).

        Raises:
            NoiseError: If input noise_ops are not a CPTP map.

        Additional Information:
            Noise ops may either be specified as list of Kraus operators
            for a general CPTP map, or as a list of `(circuit, p)` pairs
            where `circuit` is a qobj circuit for the noise, and `p` is
            the probability of the error circuit. If the input is Kraus
            operators they will be converted to the circuit format, with
            checks applied for determining if any Kraus operators are
            unitary matrices.

        Example:
            An example noise_ops for a bit-flip error with error probability
            p = 0.1 is:
            ```
            noise_ops = [([{"name": "id", "qubits": 0}], 0.9),
                         ([{"name": "x", "qubits": 0}], 0.1)]
            ```
            The same error represented as a Kraus channel can be input as:
            ```
            noise_ops = [np.sqrt(0.9) * np.array([[1, 0], [0, 1]]),
                         np.sqrt(0.1) * np.array([[0, 1], [1, 0]])]
            ```
        """

        # Initialize internal variables
        self._number_of_qubits = None
        self._noise_circuits = []
        self._noise_probabilities = []

        # Convert QuantumChannel subclass into Kraus list
        if issubclass(noise_ops.__class__, QuantumChannel):
            noise_ops, other = Kraus(noise_ops)._data
            if other is not None:
                # A Kraus map with different left and right Kraus matrices
                # cannot be CPTP
                raise NoiseError("Input QuantumChannel subclass is not CPTP.")

        # Convert iterable input into list
        noise_ops = list(noise_ops)
        # Check if Kraus
        if isinstance(noise_ops[0], np.ndarray):
            noise_ops = kraus2instructions(
                noise_ops, standard_gates, atol=atol)
        minimum_qubits = 0
        # Add non-zero probability error circuits to the error
        for circuit, prob in noise_ops:
            if prob > atol:
                self._noise_circuits.append(circuit)
                self._noise_probabilities.append(prob)
                # Determinine minimum qubit number for error from circuits
                for gate in circuit:
                    gate_qubits = max(gate["qubits"]) + 1
                    minimum_qubits = max([minimum_qubits, gate_qubits])

        # Set number of qubits
        if number_of_qubits is None:
            self._number_of_qubits = minimum_qubits
        else:
            self._number_of_qubits = number_of_qubits

        # Error checking
        if minimum_qubits > self._number_of_qubits:
            raise NoiseError("Input errors require {} qubits, "
                             "but number_of_qubits is {}".format(
                                 minimum_qubits, number_of_qubits))
        if len(self._noise_circuits) != len(self._noise_probabilities):
            raise NoiseError(
                "Number of error circuits does not match length of probabilities"
            )
        total_probs = np.sum(self._noise_probabilities)
        if abs(total_probs - 1) > atol:
            raise NoiseError(
                "Probabilities are not normalized: {} != 1".format(
                    total_probs))
        if [p for p in self._noise_probabilities if p < 0]:
            raise NoiseError("Probabilities are invalid.")

    def __repr__(self):
        """Display QuantumError."""
        return "QuantumError({})".format(
            list(zip(self.circuits, self.probabilities)))

    def __str__(self):
        """Print error information."""
        output = "QuantumError on {} qubits. Noise circuits:".format(
            self._number_of_qubits)
        for j, pair in enumerate(zip(self.probabilities, self.circuits)):
            output += "\n  P({0}) = {1}, QasmQobjInstructions = [{2}".format(j, pair[0], pair[1])
        return output

    def copy(self):
        """Make a copy of current QuantumError."""
        # pylint: disable=no-value-for-parameter
        # The constructor of subclasses from raw data should be a copy
        return copy.deepcopy(self)

    @property
    def atol(self):
        """The absolute tolerence parameter for float comparisons."""
        return QuantumError.ATOL

    @atol.setter
    def atol(self, atol):
        """Set the absolute tolerence parameter for float comparisons."""
        max_tol = QuantumChannel.MAX_TOL
        if atol < 0:
            raise NoiseError("Invalid atol: must be non-negative.")
        if atol > max_tol:
            raise NoiseError(
                "Invalid atol: must be less than {}.".format(max_tol))
        QuantumError.ATOL = atol

    @property
    def rtol(self):
        """The relative tolerence parameter for float comparisons."""
        return QuantumError.RTOL

    @rtol.setter
    def rtol(self, rtol):
        """Set the relative tolerence parameter for float comparisons."""
        max_tol = QuantumError.MAX_TOL
        if rtol < 0:
            raise NoiseError("Invalid rtol: must be non-negative.")
        if rtol > max_tol:
            raise NoiseError(
                "Invalid rtol: must be less than {}.".format(max_tol))
        QuantumChannel.RTOL = rtol

    @property
    def size(self):
        """Return the number of error circuit."""
        return len(self._noise_circuits)

    @property
    def number_of_qubits(self):
        """Return the number of qubits for the error."""
        return self._number_of_qubits

    @property
    def circuits(self):
        """Return the list of error circuits."""
        return self._noise_circuits

    @property
    def probabilities(self):
        """Return the list of error probabilities."""
        return self._noise_probabilities

    def ideal(self):
        """Return True if current error object is an identity"""
        instructions, probability = self.error_term(0)
        if probability == 1 and instructions == [{
                "name": "id",
                "qubits": [0]
        }]:
            logger.debug("Error object is ideal")
            return True
        return False

    def error_term(self, position):
        """
        Return a single term from the error.

        Args:
            position (int): the position of the error term.

        Returns:
            tuple: A pair `(p, circuit)` for error term at `position` < size
            where `p` is the probability of the error term, and `circuit`
            is the list of qobj instructions for the error term.

        Raises:
            NoiseError: If the position is greater than the size of
            the quantum error.
        """
        if position < self.size:
            return self.circuits[position], self.probabilities[position]
        else:
            raise NoiseError("Position {} is greater than the number".format(
                position) + "of error outcomes {}".format(self.size))

    def as_dict(self):
        """Return the current error as a dictionary."""
        error = {
            "type": "qerror",
            "operations": [],
            "instructions": list(self._noise_circuits),
            "probabilities": list(self._noise_probabilities)
        }
        return error

    def compose(self, other, inplace=False, front=False):
        """Return the composition error channel self∘other.

        Args:
            other (QuantumError): a quantum error channel
            inplace (bool): If True modify the current object inplace
                            [Default: False]
            front (bool): If False compose in standard order other(self(input))
                          otherwise compose in reverse order self(other(input))
                          [default: False]

        Returns:
            QuantumError: The composition error channel.

        Raises:
            NoiseError: if other is not a QuantumError, QuantumChannel subclass,
            or has incompatible dimensions.
        """
        # Convert QuantumChannel into a quantum error
        if issubclass(other.__class__, QuantumChannel):
            other = QuantumError(other)
        # Error checking
        if not isinstance(other, QuantumError):
            raise NoiseError("error1 is not a QuantumError")
        if self.number_of_qubits != other.number_of_qubits:
            raise NoiseError(
                "QuantumErrors are not defined on same number of qubits.")

        combined_noise_circuits = []
        combined_noise_probabilities = []

        # Combine subcircuits and probabilities
        if front:
            noise_ops0 = list(
                zip(other._noise_circuits, other._noise_probabilities))
            noise_ops1 = list(
                zip(self._noise_circuits, self._noise_probabilities))
        else:
            noise_ops0 = list(
                zip(self._noise_circuits, self._noise_probabilities))
            noise_ops1 = list(
                zip(other._noise_circuits, other._noise_probabilities))

        # Combine subcircuits and probabilities
        for circuit0, prob0 in noise_ops0:
            for circuit1, prob1 in noise_ops1:
                combined_noise_probabilities.append(prob0 * prob1)
                tmp_combined = circuit0 + circuit1

                # Fuse compatible ops to reduce noise operations:
                combined_circuit = [tmp_combined[0]]
                for instr in tmp_combined[1:]:
                    last_instr = combined_circuit[-1]
                    name = instr['name']
                    if name == 'id':
                        # Pass identity operation
                        pass
                    if (name == 'unitary' and last_instr['name'] == 'unitary'
                            and instr['qubits'] == last_instr['qubits']):
                        # Combine unitary matrix operations
                        combined_circuit[-1] = self._compose_unitary(
                            last_instr, instr)
                    elif (name == 'kraus' and last_instr['name'] == 'kraus'
                          and instr['qubits'] == last_instr['qubits']):
                        # Combine Kraus operations
                        combined_circuit[-1] = self._compose_kraus(
                            last_instr, instr)
                    else:
                        # Append the operation
                        combined_circuit.append(instr)
                # Check if circuit is empty and add identity
                if not combined_circuit:
                    combined_circuit.append({'name': 'id', 'qubits': [0]})
                # Add circuit
                combined_noise_circuits.append(combined_circuit)
        noise_ops = zip(combined_noise_circuits, combined_noise_probabilities)
        # Initialize new list of noise ops
        tmp = QuantumError(noise_ops)
        if inplace:
            self._number_of_qubits = tmp._number_of_qubits
            self._noise_circuits = tmp._noise_circuits
            self._noise_probabilities = tmp._noise_probabilities
            return self
        return tmp

    def power(self, n, inplace=False):
        """Return the compose of a error channel with itself n times.

        Args:
            n (int): the number of times to compose with self (n>0).
            inplace (bool): If True modify the current object inplace
                            [Default: False]

        Returns:
            QuantumError: the n-times composition error channel.

        Raises:
            NoiseError: if the power is not a positive integer.
        """
        if not isinstance(n, int) or n < 1:
            raise NoiseError("Can only power with positive integer powers.")
        # Update inplace
        if inplace:
            if n == 1:
                return self
            # cache current state to apply n-times
            cache = self.copy()
            for _ in range(1, n):
                self.compose(cache, inplace=True)
            return self
        # Return new object
        ret = self.copy()
        for _ in range(1, n):
            ret = ret.compose(self)
        return ret

    def tensor(self, other, inplace=False):
        """Return the tensor product quantum error channel self ⊗ other.

        Args:
            other (QuantumError): a quantum error channel.
            inplace (bool): If True modify the current object inplace
                            [Default: False]

        Returns:
            QuantumError: the tensor product error channel self ⊗ other.

        Raises:
            NoiseError: if other is not a QuantumError or QuantumChannel subclass.
        """
        return self._tensor_product(other, inplace=inplace, reverse=False)

    def expand(self, other, inplace=False):
        """Return the tensor product quantum error channel self ⊗ other.

        Args:
            other (QuantumError): a quantum error channel.
            inplace (bool): If True modify the current object inplace
                            [Default: False]

        Returns:
            QuantumError: the tensor product error channel other ⊗ self.

        Raises:
            NoiseError: if other is not a QuantumError or QuantumChannel subclass.
        """
        return self._tensor_product(other, inplace=inplace, reverse=True)

    def kron(self, other):
        """Return the tensor product quantum error channel self ⊗ other.

        DEPRECIATED: use QuantumError.tensor instead.

        Args:
            other (QuantumError, QuantumChannel): a quantum error or channel.

        Returns:
            QuantumError: the tensor product channel self ⊗ other.
        """
        warnings.warn(
            'The kron() method is deprecated and will be removed '
            'in a future release. Use QuantumError.tensor instead.',
            DeprecationWarning)
        return self.tensor(other)

    def _tensor_product(self, other, inplace=False, reverse=False):
        """Return the tensor product error channel.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                            [default: False]
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False
        Returns:
            QuantumError: the tensor product error channel.

        Raises:
            NoiseError: if other is not a QuantumError or QuantumChannel subclass.
        """
        # Convert QuantumChannel into a quantum error
        if issubclass(other.__class__, QuantumChannel):
            other = QuantumError(other)
        # Error checking
        if not isinstance(other, QuantumError):
            raise NoiseError("{} is not a QuantumError".format(other))

        combined_noise_circuits = []
        combined_noise_probabilities = []
        # Combine subcircuits and probabilities
        if reverse:
            shift_qubits = other.number_of_qubits
            noise_ops0 = list(
                zip(self._noise_circuits, self._noise_probabilities))
            noise_ops1 = list(
                zip(other._noise_circuits, other._noise_probabilities))
        else:
            shift_qubits = self.number_of_qubits
            noise_ops0 = list(
                zip(other._noise_circuits, other._noise_probabilities))
            noise_ops1 = list(
                zip(self._noise_circuits, self._noise_probabilities))
        for circuit1, prob1 in noise_ops1:
            for circuit0, prob0 in noise_ops0:
                combined_noise_probabilities.append(prob0 * prob1)
                # Shift qubits in circuit1
                circuit1_shift = []
                for instr in circuit1:
                    tmp = instr.copy()
                    tmp['qubits'] = [q + shift_qubits for q in tmp['qubits']]
                    circuit1_shift.append(tmp)
                tmp_combined = circuit0 + circuit1_shift
                # Fuse compatible ops to reduce noise operations:
                combined_circuit = [tmp_combined[0]]
                for instr in tmp_combined[1:]:
                    last_instr = combined_circuit[-1]
                    name = instr['name']
                    if name == 'unitary' and last_instr['name'] == 'unitary':
                        if self._qubits_distinct(last_instr['qubits'],
                                                 instr['qubits']):
                            # Combine unitary matrix operations
                            combined_circuit[-1] = self._tensor_unitary(
                                instr, last_instr)
                    elif (name == 'kraus' and last_instr['name'] == 'kraus'
                          and self._qubits_distinct(last_instr['qubits'],
                                                    instr['qubits'])):
                        # Combine Kraus operations
                        combined_circuit[-1] = self._tensor_kraus(
                            instr, last_instr)
                    elif name != 'id':
                        # Append the operation
                        combined_circuit.append(instr)
                # Check if circuit is empty and add identity
                if not combined_circuit:
                    combined_circuit.append({'name': 'id', 'qubits': [0]})
                # Add circuit
                combined_noise_circuits.append(combined_circuit)
        noise_ops = zip(combined_noise_circuits, combined_noise_probabilities)
        # Initialize new list of noise ops
        tmp = QuantumError(noise_ops)
        if inplace:
            self._number_of_qubits = tmp._number_of_qubits
            self._noise_circuits = tmp._noise_circuits
            self._noise_probabilities = tmp._noise_probabilities
            return self
        return tmp

    @staticmethod
    def _tensor_kraus(kraus1, kraus0):
        """Helper function for tensor of two kraus qobj instructions."""
        qubits = kraus0['qubits'] + kraus1['qubits']
        params = [
            np.kron(b, a) for a in kraus0['params'] for b in kraus1['params']
        ]
        return {'name': 'kraus', 'qubits': qubits, 'params': params}

    @staticmethod
    def _tensor_unitary(unitary1, unitary0):
        """Helper function for tensor of two unitary qobj instructions."""
        qubits = unitary0['qubits'] + unitary1['qubits']
        params = np.kron(unitary1['params'], unitary0['params'])
        return {'name': 'unitary', 'qubits': qubits, 'params': params}

    @staticmethod
    def _compose_kraus(kraus0, kraus1):
        """Helper function for compose of two kraus qobj instructions."""
        qubits0 = kraus0['qubits']
        qubits1 = kraus1['qubits']
        if qubits0 != qubits1:
            raise NoiseError("Kraus instructions are on different qubits")
        params = [
            np.dot(b, a) for a in kraus0['params'] for b in kraus1['params']
        ]
        return {'name': 'kraus', 'qubits': qubits0, 'params': params}

    @staticmethod
    def _compose_unitary(mat0, mat1):
        """Helper function for compose of two unitary qobj instructions."""
        qubits0 = mat0['qubits']
        qubits1 = mat1['qubits']
        if qubits0 != qubits1:
            raise NoiseError("Unitary instructions are on different qubits")
        params = np.dot(mat1['params'], mat0['params'])
        return {'name': 'unitary', 'qubits': qubits0, 'params': params}

    @staticmethod
    def _qubits_distinct(qubits0, qubits1):
        """Return true if two lists of qubits are distinct."""
        joint = qubits0 + qubits1
        return len(set(joint)) == len(joint)
