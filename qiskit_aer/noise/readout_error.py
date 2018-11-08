# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Readout error class for Qiskit Aer noise model.
"""

from .noise_utils import qubits_from_mat


class ReadoutError:
    """
    Readout error class for Qiskit Aer noise model.
    """

    def __init__(self, noise_op):
        """
        Create a readout error for a noise model.

        Args:
            noise_op (array like): An assignment fidelity probability matrix..

        Additional Information:
            The assignment fidelity probability matrix must have rows
            corresponding to probability vectors.
        """

        # TODO: Check input probability matrix is valid
        self._probabilities = noise_op
        self._number_of_qubits = qubits_from_mat(noise_op)

    @property
    def number_of_qubits(self):
        """Return the number of qubits for the error."""
        return self._number_of_qubits

    @property
    def probabilities(self):
        """Return the readout error probabilities matrix."""
        return self._probabilities

    def as_dict(self):
        """Return the current error as a dictionary."""
        error = {"type": "roerror",
                 "operations": ["measure"],
                 "probabilities": self._probabilities}
        return error

    def __repr__(self):
        """Display ReadoutError."""
        return "ReadoutError({})".format(self._probabilities)

    def __str__(self):
        """Print error information."""
        output = "ReadoutError on {} qubits.".format(self._number_of_qubits) + \
                 " Assignment probabilities:"
        for j, vec in enumerate(self._probabilities):
            output += "\n P(j|{0}) =  {1}".format(j, vec)
        return output
