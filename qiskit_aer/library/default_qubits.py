# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Helper function
"""

from qiskit.circuit import QuantumRegister


def default_qubits(circuit, qubits=None):
    """Helper method to return list of qubits.

    Args:
        circuit (QuantumCircuit): a quantum circuit.
        qubits (list or QuantumRegister): Optional, qubits argument,
            If None the returned list will be all qubits in the circuit.
            [Default: None]

    Raises:
        ValueError: if default qubits fails.

    Returns:
        list: qubits list.
    """
    # Convert label to string for backwards compatibility
    # If no qubits are specified we add all qubits so it acts as a barrier
    # This is needed for full register snapshots like statevector
    if isinstance(qubits, QuantumRegister):
        qubits = qubits[:]
    if qubits is None:
        qubits = list(circuit.qubits)
        if len(qubits) == 0:
            raise ValueError("no qubits for snapshot")

    return qubits
