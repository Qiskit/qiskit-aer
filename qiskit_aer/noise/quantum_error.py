# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Quantum error class for Qiskit Aer noise model
"""

import numpy as np
from .aernoiseerror import AerNoiseError
from .noise_utils import (kraus2instructions, mat_dot, kraus_dot,
                          mat_kron, kraus_kron, qubits_distinct)


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
            threshold (double): The threshold parameter for testing if
                                probabilities are normalized and Kraus
                                operators are unitary (default: 1e-10).

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

        # Check if Kraus
        if isinstance(noise_ops, (list, tuple)) and isinstance(noise_ops[0], np.ndarray):
            noise_ops = kraus2instructions(noise_ops,
                                           standard_gates=standard_gates,
                                           threshold=threshold)

        minimum_qubits = 0
        # Add non-zero probability error circuits to the error
        for circuit, prob in noise_ops:
            if prob > 0:
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
            raise AerNoiseError("Input errors require {} qubits, ".format(minimum_qubits) +
                                "but number_of_qubits is {}".format(number_of_qubits))
        if len(self._noise_circuits) != len(self._noise_probabilities):
            raise AerNoiseError("Number of error circuits does not match length of probabilities")
        total_probs = np.sum(self._noise_probabilities)
        if abs(total_probs - 1) > threshold:
            raise AerNoiseError("Probabilities are not normalized: {} != 1".format(total_probs))
        if len([p for p in self._noise_probabilities if p < 0]) > 0:
            raise AerNoiseError("Probabilities are invalid.")

    def __repr__(self):
        """Display QuantumError."""
        return "QuantumError({})".format(list(zip(self.probabilities, self.circuits)))

    def __str__(self):
        """Print error information."""
        output = "QuantumError on {} qubits. Noise circuits:".format(self._number_of_qubits)
        for j, pair in enumerate(zip(self.probabilities, self.circuits)):
            output += "\n  P({0}) = {1}, circuit = [{2}".format(j, pair[0], pair[1])
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
            AerNoiseError: If the position is greater than the size of
            the quantum error.
        """
        if position < self.size:
            return self.circuits[position], self.probabilities[position]
        else:
            raise AerNoiseError("Position {} is greater than the number".format(position) +
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
            raise AerNoiseError("error1 is not a QuantumError")
        if self.number_of_qubits != error.number_of_qubits:
            raise AerNoiseError("QuantumErrors are not defined on same number of qubits.")

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
                    if (name == 'mat' and last_op['name'] == 'mat' and
                            op['qubits'] == last_op['qubits']):
                        # Combine unitary matrix operations
                        combined_circuit[-1] = mat_dot(last_op, op)
                    elif (name == 'kraus' and last_op['name'] == 'kraus' and
                          op['qubits'] == last_op['qubits']):
                        # Combine Kraus operations
                        combined_circuit[-1] = kraus_dot(last_op, op)
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
            second error will be shifted by the number of qubits in the
            first error.
        """

        # Error checking
        if not isinstance(error, QuantumError):
            raise AerNoiseError("{} is not a QuantumError".format(error))

        combined_noise_circuits = []
        combined_noise_probabilities = []

        # Combine subcircuits and probabilities
        shift_qubits = self.number_of_qubits
        for circuit0, prob0 in zip(self._noise_circuits, self._noise_probabilities):
            for circuit1, prob1 in zip(error._noise_circuits, error._noise_probabilities):
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
                    if (name == 'mat' and last_op['name'] == 'mat' and
                            qubits_distinct(last_op['qubits'], op['qubits'])):
                        # Combine unitary matrix operations
                        combined_circuit[-1] = mat_kron(last_op, op)
                    elif (name == 'kraus' and last_op['name'] == 'kraus' and
                          qubits_distinct(last_op['qubits'], op['qubits'])):
                        # Combine Kraus operations
                        combined_circuit[-1] = kraus_kron(last_op, op)
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
