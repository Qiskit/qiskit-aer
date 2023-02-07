# This code is part of Qiskit.
#
# (C) Copyright IBM 2017-2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from abc import ABC
from typing import List, Dict, Optional, Tuple
import numpy as np

from qiskit.circuit import QuantumCircuit, CircuitInstruction, Parameter, ParameterExpression
from qiskit.qobj import QasmQobjInstruction
from qiskit_aer.backends.controller_wrappers import AerCircuit_, AerOp_, OpType_, DataSubType_, create_aer_operation
from qiskit_aer.aererror import AerError

"""Directly simulatable circuit operation in Aer."""
class AerOp:
    """A class of an internal ciruict operation of Aer
    """
    def __init__(
        self,
        qobj_inst: Optional[QasmQobjInstruction]
    ):
        self._conditional = False
        self._conditional_reg = -1
        self._qobj_inst = qobj_inst
        if hasattr(qobj_inst, 'params'):
            for i in range(len(qobj_inst.params)):
                if (isinstance(qobj_inst.params[i], ParameterExpression)
                      and len(qobj_inst.params[i].parameters) > 0):
                    qobj_inst.params[i] = 0.0
    
    def set_conditional(
        self,
        conditional_reg: int
    )-> None:
        self._conditional = conditional_reg >= 0
        self._conditional_reg = conditional_reg

    def assemble_native(
        self
    )-> AerOp_:
        op = create_aer_operation(self._qobj_inst)
        op.conditional = self._conditional
        if self._conditional:
            op.conditional_reg = self._conditional_reg
        return op

class BinaryFuncOp(AerOp):
    """bfunc operation"""
    def __init__(
        self,
        mask: str,
        relation: str,
        val: str,
        conditional_reg_idx: int
    ):
        super().__init__(
            QasmQobjInstruction(name="bfunc",
                                mask="0x%X" % mask,
                                relation="==",
                                val="0x%X" % val,
                                register=conditional_reg_idx,
                                )
            )

def generate_aer_operation(
    inst: CircuitInstruction,
    qubit_indices: Dict[int, int],
    clbit_indices: Dict[int, int],
    register_clbits: bool
) -> AerOp:

    qobj_inst = inst.operation.assemble()

    if inst.qubits:
        qobj_inst.qubits = [qubit_indices[qubit] for qubit in inst.qubits]
    if inst.clbits:
        qobj_inst.memory = [clbit_indices[clbit] for clbit in inst.clbits]
        if inst.operation.name == "measure" and register_clbits:
            qobj_inst.register = [clbit_indices[clbit] for clbit in inst.clbits]

    return AerOp(qobj_inst=qobj_inst)