# This code is part of Qiskit.
#
# (C) Copyright IBM 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Simulator command to perform multiple pauli gates in a single pass
"""
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import XGate, YGate, ZGate
from qiskit.circuit.gate import Gate


class MultiPauliGate(Gate):
    r"""A multi-qubit Pauli gate.

        This gate exists for optimization purposes for the
        quantum statevector simulation, since applying multiple
        pauli gates to different qubits at once can be done via
        a single pass on the statevector.

        The functionality is equivalent to applying
        the pauli gates sequentially using standard Qiskit gates
        """

    def __init__(self, pauli_string=None):
        self.pauli_string = pauli_string
        super().__init__('multi_pauli', len(pauli_string), [pauli_string[::-1]])

    def _define(self):
        """
        gate multi_pauli (p1 a1,...,pn an) { p1 a1; ... ; pn an; }
        """
        gates = {'X': XGate, 'Y': YGate, 'Z': ZGate}
        q = QuantumRegister(len(self.pauli_string), 'q')
        qc = QuantumCircuit(q, name=self.name)

        rules = [(gates[p](), [q[i]], []) for (i, p) in enumerate(self.pauli_string)]
        qc._data = rules
        self.definition = qc

    def inverse(self):
        r"""Return inverted multi-pauli gate (itself)."""
        return MultiPauliGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the multi-pauli gate.
        i.e. tensor product of the paulis"""
        pauli_matrices = {
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex)
        }
        mat = np.eye(1, dtype=complex)
        for pauli in self.pauli_string:
            mat = np.kron(mat, pauli_matrices[pauli])
        return mat


def multi_pauli(self, qubits, pauli_string):
    """Adds a multi pauli instruction to the circuit"""
    return self.append(MultiPauliGate(pauli_string), qubits)


# Add to QuantumCircuit class
QuantumCircuit.multi_pauli = multi_pauli
