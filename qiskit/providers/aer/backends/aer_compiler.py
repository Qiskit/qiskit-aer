# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Compier to convert Qiskit control-flow to Aer backend.
"""
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.controlflow import (
    WhileLoopOp,
    ForLoopOp,
    IfElseOp,
    BreakLoopOp,
    ContinueLoopOp)


class AerMark(Instruction):
    """
    Mark instruction

    This instruction is a destination of jump instruction.
    Conditional is not allowed in Aer controller.
    """

    def __init__(self, name, num_qubits):
        super().__init__("mark", num_qubits, 0, [name])


class AerJump(Instruction):
    """
    Jump instruction

    This instruction sets a program counter to specified mark instruction.
    """

    def __init__(self, jump_to, num_qubits):
        super().__init__("jump", num_qubits, 0, [jump_to])


class AerCompiler:
    """
    Aer Compiler

    Convert instructions of control flow to mark and jump instructions
    """

    def __init__(self):
        self._last_label_id = -1

    def _inline_for_loop_op(self, inst):
        loop_parameter, indexset, body = inst.params

        self._last_label_id += 1
        loop_id = self._last_label_id
        loop_name = f'loop_{loop_id}'

        ret = QuantumCircuit()
        for qr in body.qregs:
            ret.add_register(qr)
        for cr in body.cregs:
            ret.add_register(cr)

        inlined_body = None
        break_label = f'{loop_name}_end'
        for index in indexset:
            continue_label = f'{loop_name}_{index}'
            inlined_body = self._inline_circuit(body.bind_parameters({loop_parameter: index}),
                                                continue_label,
                                                break_label)
            ret.append(inlined_body,
                       range(inlined_body.num_qubits),
                       range(inlined_body.num_clbits))
            ret.append(AerMark(continue_label, inlined_body.num_qubits),
                       range(inlined_body.num_qubits), [])

        if inlined_body:
            ret.append(AerMark(break_label, inlined_body.num_qubits),
                       range(inlined_body.num_qubits), [])
        return ret

    def _inline_while_loop_op(self, inst):
        condition, body = inst.params

        self._last_label_id += 1
        loop_id = self._last_label_id
        loop_name = f'while_{loop_id}'

        ret = QuantumCircuit()
        for qr in body.qregs:
            ret.add_register(qr)
        for cr in body.cregs:
            ret.add_register(cr)

        continue_label = f'{loop_name}_start'
        break_label = f'{loop_name}_end'
        inlined_body = self._inline_circuit(body, continue_label, break_label)
        ret.append(inlined_body,
                   range(inlined_body.num_qubits),
                   range(inlined_body.num_clbits))
        ret.append(AerJump(continue_label, ret.num_qubits),
                   range(ret.num_qubits), [])
        ret.append(AerMark(break_label, inlined_body.num_qubits),
                   range(inlined_body.num_qubits), [])
        return condition, continue_label, break_label, ret

    def _inline_if_else_op(self, inst, continue_label, break_label):
        condition, true_body, false_body = inst.params

        self._last_label_id += 1
        if_id = self._last_label_id
        if_name = f'if_{if_id}'

        ret = QuantumCircuit()
        for qr in true_body.qregs:
            ret.add_register(qr)
        for cr in true_body.cregs:
            ret.add_register(cr)

        if_else_label = f'{if_name}_else'
        if_end_label = f'{if_name}_end'

        ret.append(self._inline_circuit(true_body, continue_label, break_label),
                   range(ret.num_qubits),
                   range(ret.num_clbits))

        if false_body:
            ret.append(AerJump(if_end_label, ret.num_qubits),
                       range(ret.num_qubits), [])

            ret.append(AerMark(if_else_label, ret.num_qubits),
                       range(ret.num_qubits), [])

            ret.append(self._inline_circuit(false_body, continue_label, break_label),
                       range(ret.num_qubits),
                       range(ret.num_clbits))

        ret.append(AerMark(if_end_label, ret.num_qubits),
                   range(ret.num_qubits), [])

        return condition, if_else_label if false_body else if_end_label, ret

    def _inline_circuit(self, circ, continue_label, break_label):
        ret = QuantumCircuit()
        for qr in circ.qregs:
            ret.add_register(qr)
        for cr in circ.cregs:
            ret.add_register(cr)

        q2i = {}
        for q in circ.qubits:
            q2i[q] = len(q2i)
        c2i = {}
        for c in circ.clbits:
            c2i[c] = len(c2i)

        for inst, qargs, cargs in circ.data:
            if isinstance(inst, ForLoopOp):
                ret.append(self._inline_for_loop_op(inst),
                           [q2i[q] for q in qargs],
                           [c2i[c] for c in cargs])
            elif isinstance(inst, WhileLoopOp):
                condition, _continue_label, _break_label, body = self._inline_while_loop_op(inst)
                ret.append(AerMark(_continue_label, ret.num_qubits),
                           range(ret.num_qubits), [])
                ret.append(AerJump(_break_label, ret.num_qubits).c_if(condition, 0),
                           range(ret.num_qubits), [])
                ret.append(body, [q2i[q] for q in qargs], [c2i[c] for c in cargs])
            elif isinstance(inst, IfElseOp):
                condition, else_label, body = self._inline_if_else_op(inst,
                                                                      continue_label,
                                                                      break_label)
                ret.append(AerJump(else_label, ret.num_qubits).c_if(condition, 0),
                           range(ret.num_qubits), [])
                ret.append(body, [q2i[q] for q in qargs], [c2i[c] for c in cargs])
            elif isinstance(inst, BreakLoopOp):
                ret.append(AerJump(break_label, ret.num_qubits),
                           range(ret.num_qubits), [])
            elif isinstance(inst, ContinueLoopOp):
                ret.append(AerJump(continue_label, ret.num_qubits),
                           range(ret.num_qubits), [])
            else:
                ret.append(inst, qargs, cargs)

        return ret

    def compile(self, circ):
        """
        compile a circuit that have control-flow instructions
        """
        return self._inline_circuit(circ, None, None)
