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

"""
Simulator command to snapshot internal simulator representation.
"""

from qiskit import QuantumCircuit
from qiskit.providers.aer.extensions import Snapshot


class SnapshotExpectationValue(Snapshot):
    """Snapshot instruction for supported methods of Qasm simulator."""

    def __init__(self,
                 label,
                 op,
                 pauli=True,
                 variance=False,
                 num_qubits=0,
                 num_clbits=0,
                 params=None):

        num_qubits = len(op)

        if pauli:
            if variance:
                super().__init__(label,
                                 'expectation_value_pauli_with_variance',
                                 num_qubits,
                                 num_clbits,
                                 params)
            else:
                super().__init__(label,
                                 'expectation_value_pauli',
                                 num_qubits,
                                 num_clbits,
                                 params)

        else:
            if variance:
                super().__init__(label, 'expectation_value_matrix_with_variance', num_qubits,
                                 num_clbits, params)
            else:
                super().__init__(label, 'expectation_value_matrix', num_qubits, num_clbits, params)


def snapshot_expectation_value(self,
                               label,
                               op,
                               qubits=None,
                               params=None):
    """Take a snapshot of expectation value <M> of some Operator M.
    Works on all qubits, and prevents reordering (like barrier).
    Args:
        label (str): a snapshot label to report the result
        op (Operator): operator to snapshot
        qubits (list or None): the qubits to apply snapshot to [Default: None].
        params (list or None): the parameters for snapshot_type [Default: None].
    Returns:
        QuantumCircuit: with attached command
    Raises:
        ExtensionError: malformed command
    """

    snapshot_register = Snapshot.define_snapshot_register(self, label, qubits)

    return self.append(
        SnapshotExpectationValue(
            label,
            op,
            params=params), snapshot_register)


QuantumCircuit.snapshot_expectation_value = snapshot_expectation_value
