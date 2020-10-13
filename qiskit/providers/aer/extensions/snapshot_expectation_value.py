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
from warnings import warn
import math
import numpy
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.extensions.exceptions import ExtensionError
from qiskit.qobj import QasmQobjInstruction
from qiskit.quantum_info.operators import Pauli, Operator
from .snapshot import Snapshot


class SnapshotExpectationValue(Snapshot):
    """Snapshot instruction for supported methods of Qasm simulator."""

    def __init__(self, label, op, single_shot=False, variance=False):
        """Create an expectation value snapshot instruction.

        Args:
            label (str): the snapshot label.
            op (Operator): operator to snapshot.
            single_shot (bool): return list for each shot rather than average [Default: False]
            variance (bool): compute variance of values [Default: False]

        Raises:
            ExtensionError: if snapshot is invalid.
        """
        if variance:
            warn('The snapshot `variance` kwarg has been deprecated and will'
                 ' be removed in qiskit-aer 0.8. To compute variance use'
                 ' `single_shot=True` and compute manually in post-processing',
                 DeprecationWarning)
        pauli_op = self._format_pauli_op(op)
        if pauli_op:
            # Pauli expectation value
            snapshot_type = 'expectation_value_pauli'
            params = pauli_op
            num_qubits = len(params[0][1])
        else:
            snapshot_type = 'expectation_value_matrix'
            mat = self._format_single_matrix(op)
            if mat is not None:
                num_qubits = int(math.log2(len(mat)))
                if mat.shape != (2 ** num_qubits, 2 ** num_qubits):
                    raise ExtensionError("Snapshot Operator is invalid.")
                qubits = list(range(num_qubits))
                params = [[1., [[qubits, mat]]]]
            else:
                # If op doesn't match the previous cases we try passing
                # in the op as raw params
                params = op
                num_qubits = 0
                for _, pair in params:
                    num_qubits = max(num_qubits, *pair[0])

        # HACK: we wrap param list in numpy array to make it validate
        # in terra
        params = [numpy.array(elt, dtype=object) for elt in params]

        if single_shot:
            snapshot_type += '_single_shot'
        elif variance:
            snapshot_type += '_with_variance'
        super().__init__(label,
                         snapshot_type=snapshot_type,
                         num_qubits=num_qubits,
                         params=params)

    @staticmethod
    def _format_single_matrix(op):
        """Format op into Matrix op, return None if not Pauli op"""
        # This can be specified as list [[coeff, Pauli], ... ]
        if isinstance(op, numpy.ndarray):
            return op
        if isinstance(op, (Instruction, QuantumCircuit)):
            return Operator(op).data
        if hasattr(op, 'to_operator'):
            return op.to_operator().data
        return None

    @staticmethod
    def _format_pauli_op(op):
        """Format op into Pauli op, return None if not Pauli op"""
        # This can be specified as list [[coeff, Pauli], ... ]
        if isinstance(op, Pauli):
            return [[1., op.to_label()]]
        if not isinstance(op, (list, tuple)):
            return None
        pauli_op = []
        for pair in op:
            if len(pair) != 2:
                return None
            coeff = complex(pair[0])
            pauli = pair[1]
            if isinstance(pauli, Pauli):
                pauli_op.append([coeff, pauli.to_label()])
            elif isinstance(pair[1], str):
                pauli_op.append([coeff, pauli])
            else:
                return None
        return pauli_op

    def assemble(self):
        """Assemble a QasmQobjInstruction for snapshot_expectation_value."""
        return QasmQobjInstruction(name=self.name,
                                   params=[x.tolist() for x in self.params],
                                   snapshot_type=self.snapshot_type,
                                   qubits=list(range(self.num_qubits)),
                                   label=self.label)


def snapshot_expectation_value(self, label, op, qubits,
                               single_shot=False,
                               variance=False):
    """Take a snapshot of expectation value <O> of an Operator.

    Args:
        label (str): a snapshot label to report the result
        op (Operator): operator to snapshot
        qubits (list): the qubits to snapshot.
        single_shot (bool): return list for each shot rather than average [Default: False]
        variance (bool): compute variance of values [Default: False]

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ExtensionError: if snapshot is invalid.
    """

    snapshot_register = Snapshot.define_snapshot_register(self, qubits=qubits)

    return self.append(
        SnapshotExpectationValue(label, op,
                                 single_shot=single_shot,
                                 variance=variance),
        snapshot_register)


QuantumCircuit.snapshot_expectation_value = snapshot_expectation_value
