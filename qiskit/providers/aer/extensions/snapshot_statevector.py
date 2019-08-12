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

from qiskit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.providers.aer.extensions import snapshot

class SnapshotStatevector(Snapshot):
    """ Simulator snapshot instruction for statevector snapshot type """

    def __init__(self,
                 label,
                 snapshot_type='statevector',
                 num_qubits=0,
                 num_clbits=0,
                 params=None):
        super().__init__(label, snapshot_type, num_qubits, num_clbits, params)


def snapshot_statevector(self,
                         label,
                         qubits=None,
                         params=None):

    qubits = Snapshot.define_snapshot_register(label, qubits)

    return self.append(
        SnapshotStatevector(
            label,
            snapshot_type='statevector',
            num_qubits=len(qubits),
            params=params), qubits)

QuantumCircuit.snapshot_statevector = snapshot_statevector
