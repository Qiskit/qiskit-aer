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
# that they have been altered from the originals

from qiskit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.providers.aer.extensions import snapshot

class SnapshotProbabilites(Snapshot):

    def __init__(self,
                 label,
                 num_qubits,
                 snapshot_type='probabilities',
                 num_clbits=0,
                 params=None,
                 variance=False):

        super().__init__(label, snapshot_type, num_qubits, num_clbits, params)

        if variance:
            snapshot_type = 'probabilities_with_variance'

def snapshot_probabilities(self,
                           label,
                           qubits=None,
                           params=None):

    # Convert label to string for backwards compatibility
    if not isinstance(label, str):
        warnings.warn(
            "Snapshot label should be a string, "
            "implicit conversion is deprecated.", DeprecationWarning)
        label = str(label)

    # If no qubits are specified we add all qubits so it acts as a barrier
    # This is needed for full register snapshots
    if isinstance(qubits, QuantumRegister):
        qubits = qubits[:]

    if not qubits:
        tuples = []
        if isinstance(self, QuantumCircuit):
            for register in self.qregs:
                tuples.append(register)
        if not tuples:
            raise ExtensionError('no qubits for snapshot')
        qubits = []
        for tuple_element in tuples:
            if isinstance(tuple_element, QuantumRegister):
                for j in range(tuple_element.size):
                    qubits.append(tuple_element[j])
            else:
                qubits.append(tuple_element)

    return self.append(
        SnapshotProbabilites(
            label,
            num_qubits=len(qubits),
            snapshot_type='probabilities',
            params=params),qubits)


QuantumCircuit.snapshot_probabilities = snapshot_probabilities
