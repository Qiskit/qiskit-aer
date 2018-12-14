# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Quantum error class for Qiskit Aer noise model
"""
import logging
import numpy as np
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

    def __init__(self, noise_ops, number_of_qubits=None,
                 standard_gates=True, threshold=1e-10):
        """
        Create a quantum error for a noise model.

        Args:
            noise_ops (list): A list of noise ops. See additional information.
            number_of_qubits (int): specify the number of qubits for the
                                    error. If None this will be determined
                                    automatically (default None).
            standard_gates (bool): Check if input matrices are standard gates.
            threshold (double): Threshold for testing if probabilities are
                                equal to 0 or 1 (Default: 1e-10).

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

        # Convert iterable input into list
        noise_ops = list(noise_ops)
        # Check if Kraus
        if isinstance(noise_ops[0], np.ndarray):
            noise_ops = kraus2instructions(noise_ops, standard_gates,
                                           threshold)
        minimum_qubits = 0
        # Add non-zero probability error circuits to the error
        for circuit, prob in noise_ops:
            if prob > threshold:
                self._noise_circuits.append(circuit)
                self._noise_probabilities.append(prob)
                # Determinine minimum qubit number for error from circuits
                for op in circuit:
                    op_qubits = max(op["qubits"]) + 1
                    minimum_qubits = max([minimum_qubits, op_qubits])

        # Set number of qubits
        if number_of_qubits is None:
            self._number_of_qubits = minimum_qubits
        else:
            self._number_of_qubits = number_of_qubits

        # Error checking
        if minimum_qubits > self._number_of_qubits:
            raise NoiseError("Input errors require {} qubits, ".format(minimum_qubits) +
                                "but number_of_qubits is {}".format(number_of_qubits))
        if len(self._noise_circuits) != len(self._noise_probabilities):
            raise NoiseError("Number of error circuits does not match length of probabilities")
        total_probs = np.sum(self._noise_probabilities)
        if abs(total_probs - 1) > threshold:
            raise NoiseError("Probabilities are not normalized: {} != 1".format(total_probs))
        if len([p for p in self._noise_probabilities if p < 0]) > 0:
            raise NoiseError("Probabilities are invalid.")

    def __repr__(self):
        """Display QuantumError."""
        return "QuantumError({})".format(list(zip(self.circuits, self.probabilities)))

    def __str__(self):
        """Print error information."""
        output = "QuantumError on {} qubits. Noise circuits:".format(self._number_of_qubits)
        for j, pair in enumerate(zip(self.probabilities, self.circuits)):
            output += "\n  P({0}) = {1}, QobjInstructions = [{2}".format(j, pair[0], pair[1])
        return output

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
        if probability == 1 and instructions == [{"name": "id", "qubits": [0]}]:
            logger.debug("Error object is ideal")
            return True
        return False

    def error_term(self, position):
        """
        Return a single term from the error.

        Args:
            position (int): the position of the error term.

        Returns:
            A pair `(p, circuit)` for error term at `position` < size
            where `p` is the probability of the error term, and `circuit`
            is the list of qobj instructions for the error term.

        Raises:
            NoiseError: If the position is greater than the size of
            the quantum error.
        """
        if position < self.size:
            return self.circuits[position], self.probabilities[position]
        else:
            raise NoiseError("Position {} is greater than the number".format(position) +
                                "of error outcomes {}".format(self.size))

    def as_dict(self):
        """Return the current error as a dictionary."""
        error = {"type": "qerror",
                 "operations": [],
                 "instructions": list(self._noise_circuits),
                 "probabilities": list(self._noise_probabilities)}
        return error

    def compose(self, error):
        """
        Compose with another quantum error.

        The resulting composite quanutm error will be equivalent to
        applying the current quantum error followed by applying the
        additional quantum error.

        Args:
            error (QuantumError): a quantum error to compose.

        Returns:
            QauntumError: the composed quantum error.

        Additional Information:
            Two quantum errors can only be composed if they apply to the
            same number of qubits.
        """
        # Error checking
        if not isinstance(error, QuantumError):
            raise NoiseError("error1 is not a QuantumError")
        if self.number_of_qubits != error.number_of_qubits:
            raise NoiseError("QuantumErrors are not defined on same number of qubits.")

        combined_noise_circuits = []
        combined_noise_probabilities = []

        # Combine subcircuits and probabilities
        for circuit0, prob0 in zip(self._noise_circuits, self._noise_probabilities):
            for circuit1, prob1 in zip(error._noise_circuits, error._noise_probabilities):
                combined_noise_probabilities.append(prob0 * prob1)
                tmp_combined = circuit0 + circuit1

                # Fuse compatible ops to reduce noise operations:
                combined_circuit = [tmp_combined[0]]
                for op in tmp_combined[1:]:
                    last_op = combined_circuit[-1]
                    name = op['name']
                    if name == 'id':
                        # Pass identity operation
                        pass
                    if (name == 'unitary' and last_op['name'] == 'unitary' and
                            op['qubits'] == last_op['qubits']):
                        # Combine unitary matrix operations
                        combined_circuit[-1] = self._compose_unitary(last_op, op)
                    elif (name == 'kraus' and last_op['name'] == 'kraus' and
                          op['qubits'] == last_op['qubits']):
                        # Combine Kraus operations
                        combined_circuit[-1] = self._compose_kraus(last_op, op)
                    else:
                        # Append the operation
                        combined_circuit.append(op)
                # Check if circuit is empty and add identity
                if len(combined_circuit) == 0:
                    combined_circuit.append({'name': 'id', 'qubits': [0]})
                # Add circuit
                combined_noise_circuits.append(combined_circuit)
        noise_ops = zip(combined_noise_circuits,
                        combined_noise_probabilities)
        return QuantumError(noise_ops)

    def kron(self, error):
        """
        Kronecker product current error with another quantum error.

        Args:
            error (QuantumError): a quantum error to compose.

        Returns:
            QauntumError: the composite quantum error.

        Additional Information:
            The resulting qauntum error will be defined on a number of
            qubits equal to the sum of both errors. The qubits of the
            current error will be shifted by the number of qubits in the
            second error.

            Example: For two single qubit errors E0 E1, E1.kron(E0) will
            be a two-qubit error where E0 acts on qubit-0 and E1 acts on
            qubit-1.
        """

        # Error checking
        if not isinstance(error, QuantumError):
            raise NoiseError("{} is not a QuantumError".format(error))

        combined_noise_circuits = []
        combined_noise_probabilities = []

        # Combine subcircuits and probabilities
        shift_qubits = error.number_of_qubits
        for circuit1, prob1 in zip(self._noise_circuits, self._noise_probabilities):
            for circuit0, prob0 in zip(error._noise_circuits, error._noise_probabilities):
                combined_noise_probabilities.append(prob0 * prob1)

                # Shift qubits in circuit1
                circuit1_shift = []
                for op in circuit1:
                    tmp = op.copy()
                    tmp['qubits'] = [q + shift_qubits for q in tmp['qubits']]
                    circuit1_shift.append(tmp)
                tmp_combined = circuit0 + circuit1_shift
                # Fuse compatible ops to reduce noise operations:
                combined_circuit = [tmp_combined[0]]
                for op in tmp_combined[1:]:
                    last_op = combined_circuit[-1]
                    name = op['name']
                    if name == 'id':
                        # Pass identity operation
                        pass
                    if (name == 'unitary' and last_op['name'] == 'unitary' and
                            self._qubits_distinct(last_op['qubits'], op['qubits'])):
                        # Combine unitary matrix operations
                        combined_circuit[-1] = self._kron_unitary(op, last_op)
                    elif (name == 'kraus' and last_op['name'] == 'kraus' and
                          self._qubits_distinct(last_op['qubits'], op['qubits'])):
                        # Combine Kraus operations
                        combined_circuit[-1] = self._kron_kraus(op, last_op)
                    else:
                        # Append the operation
                        combined_circuit.append(op)
                # Check if circuit is empty and add identity
                if len(combined_circuit) == 0:
                    combined_circuit.append({'name': 'id', 'qubits': [0]})
                # Add circuit
                combined_noise_circuits.append(combined_circuit)
        noise_ops = zip(combined_noise_circuits,
                        combined_noise_probabilities)
        return QuantumError(noise_ops)

    @staticmethod
    def _kron_kraus(kraus1, kraus0):
        """
        Helper function for kron of two kraus qobj instructions.
        """
        qubits = kraus0['qubits'] + kraus1['qubits']
        params = [np.kron(b, a) for a in kraus0['params']
                  for b in kraus1['params']]
        return {'name': 'kraus', 'qubits': qubits, 'params': params}

    @staticmethod
    def _kron_unitary(unitary1, unitary0):
        """
        Helper function for kron of two unitary qobj instructions.
        """
        qubits = unitary0['qubits'] + unitary1['qubits']
        params = np.kron(unitary1['params'], unitary0['params'])
        return {'name': 'unitary', 'qubits': qubits, 'params': params}

    @staticmethod
    def _compose_kraus(kraus0, kraus1):
        qubits0 = kraus0['qubits']
        qubits1 = kraus1['qubits']
        if qubits0 != qubits1:
            raise NoiseError("Kraus instructions are on different qubits")
        params = [np.dot(b, a) for a in kraus0['params']
                  for b in kraus1['params']]
        return {'name': 'kraus', 'qubits': qubits0, 'params': params}

    @staticmethod
    def _compose_unitary(mat0, mat1):
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
