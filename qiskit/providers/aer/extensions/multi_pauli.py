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

from qiskit import QuantumCircuit
from qiskit.circuit.gate import Gate

class MultiPauliGate(Gate):
    def __init__(self, pauli_string):
        super().__init__('multi_pauli', len(pauli_string), [pauli_string])

def multi_pauli(self, qubits, pauli_string):
    return self.append(MultiPauliGate(pauli_string), qubits)

# Add to QuantumCircuit class
QuantumCircuit.multi_pauli = multi_pauli