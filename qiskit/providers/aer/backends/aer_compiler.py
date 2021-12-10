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
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.controlflow import (
    WhileLoopOp,
    ForLoopOp,
    IfElseOp,
    BreakLoopOp,
    ContinueLoopOp)
from qiskit.compiler import transpile
from ..library.control_flow_instructions import AerMark, AerJump

class AerCompiler:
    """
    Aer Compiler
    Convert instructions of control flow to mark and jump instructions
    """

    def __init__(self, backend):
        self._last_flow_id = -1
        self._backend_configuration = backend.configuration()

    def compile(self, circuits):
        """
        compile a circuit that have control-flow instructions
        """
        if isinstance(circuits, list):
            all_static = all([not self._is_dynamic(circuit) for circuit in circuits])
            if all_static:
                return circuits
            basis_gates = self._backend_configuration.basis_gates + ['mark', 'jump']
            return [transpile(self._inline_circuit(self._decompose_subcircuits(circuit),
                                                   None, None),
                              basis_gates=basis_gates) if self._is_dynamic(circuit)
                    else circuit for circuit in circuits]
        else:
            return (transpile(self._inline_circuit(self._decompose_subcircuits(circuits),
                                                   None, None),
                              basis_gates=basis_gates) if self._is_dynamic(circuits)
                    else circuits)

    def _is_dynamic(self, circuit):
        if not isinstance(circuit, QuantumCircuit):
            return False
        for inst, _, _ in circuit.data:
            if isinstance(inst, (WhileLoopOp, ForLoopOp, IfElseOp, BreakLoopOp, ContinueLoopOp)):
                return True
        return False

    def _decompose_subcircuits(self, circuit, basis_gates=None):
        if circuit is None:
            return None
        if basis_gates is None:
            basis_gates = self._backend_configuration.basis_gates + ['for_loop', 'while_loop',
                                                                     'if_else', 'break_loop',
                                                                     'continue_loop']
        circuit = circuit.copy()
        for inst, _, _ in circuit.data:
            if isinstance(inst, WhileLoopOp):
                body, = inst.params
                inst.params = (self._decompose_subcircuits(body, basis_gates),)
            elif isinstance(inst, ForLoopOp):
                loop_parameter, indexset, body = inst.params
                inst.params = (loop_parameter, indexset,
                               self._decompose_subcircuits(body, basis_gates))
            elif isinstance(inst, IfElseOp):
                true_body, false_body = inst.params
                inst.params = (self._decompose_subcircuits(true_body, basis_gates),
                               self._decompose_subcircuits(false_body, basis_gates))
        circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=0)
        return circuit

    def _inline_circuit(self, circ, continue_label, break_label):
        ret = circ.copy()
        ret.data = []

        q2i = {}
        for q in ret.qubits:
            q2i[q] = len(q2i)
        c2i = {}
        for c in ret.clbits:
            c2i[c] = len(c2i)

        for inst, qargs, cargs in circ.data:
            if isinstance(inst, ForLoopOp):
                ret.append(self._inline_for_loop_op(inst),
                           [q2i[q] for q in qargs],
                           [c2i[c] for c in cargs])
            elif isinstance(inst, WhileLoopOp):
                (cond_tuple, continue_label,
                 loop_start_label, break_label, body) = self._inline_while_loop_op(inst)
                c_if_args = self._convert_c_if_args(cond_tuple)
                ret.append(AerMark(continue_label, ret.num_qubits),
                           range(ret.num_qubits), [])
                ret.append(AerJump(loop_start_label, ret.num_qubits).c_if(*c_if_args),
                           range(ret.num_qubits), [])
                ret.append(AerJump(break_label, ret.num_qubits),
                           range(ret.num_qubits), [])
                ret.append(AerMark(loop_start_label, ret.num_qubits),
                           range(ret.num_qubits), [])
                ret.append(body, [q2i[q] for q in qargs], [c2i[c] for c in cargs])
            elif isinstance(inst, IfElseOp):
                cond_tuple, true_label, else_label, body = self._inline_if_else_op(inst,
                                                                                   continue_label,
                                                                                   break_label)
                c_if_args = self._convert_c_if_args(cond_tuple)
                ret.append(AerJump(true_label, ret.num_qubits).c_if(*c_if_args),
                           range(ret.num_qubits), [])
                ret.append(AerJump(else_label, ret.num_qubits),
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

    def _convert_c_if_args(self, cond_tuple):
        return [1 if elem is True else 0 if elem is False else elem for elem in cond_tuple]

    def _inline_for_loop_op(self, inst):
        indexset, loop_parameter, body = inst.params

        self._last_flow_id += 1
        loop_id = self._last_flow_id
        loop_name = f'loop_{loop_id}'

        ret = QuantumCircuit()
        for qr in body.qregs:
            ret.add_register(qr)
        for cr in body.cregs:
            ret.add_register(cr)
        if len(ret.clbits) == 0:
            ret.add_bits(body.clbits)
        if len(ret.qubits) == 0:
            ret.add_bits(body.qubits)

        inlined_body = None
        break_label = f'{loop_name}_end'
        for index in indexset:
            continue_label = f'{loop_name}_{index}'
            inlined_body = self._inline_circuit(body,
                                                continue_label,
                                                break_label)
            inlined_body = inlined_body.bind_parameters({loop_parameter: index})
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
        condition_tuple = inst.condition
        body, = inst.params

        self._last_flow_id += 1
        loop_id = self._last_flow_id
        loop_name = f'while_{loop_id}'

        ret = QuantumCircuit()
        for qr in body.qregs:
            ret.add_register(qr)
        for cr in body.cregs:
            ret.add_register(cr)
        if len(ret.clbits) == 0:
            ret.add_bits(body.clbits)
        if len(ret.qubits) == 0:
            ret.add_bits(body.qubits)

        continue_label = f'{loop_name}_continue'
        loop_start_label = f'{loop_name}_start'
        break_label = f'{loop_name}_end'
        inlined_body = self._inline_circuit(body, continue_label, break_label)
        ret.append(inlined_body,
                   range(inlined_body.num_qubits),
                   range(inlined_body.num_clbits))
        ret.append(AerJump(continue_label, ret.num_qubits),
                   range(ret.num_qubits), [])
        ret.append(AerMark(break_label, inlined_body.num_qubits),
                   range(inlined_body.num_qubits), [])
        return condition_tuple, continue_label, loop_start_label, break_label, ret

    def _inline_if_else_op(self, inst, continue_label, break_label):
        condition_tuple = inst.condition
        true_body, false_body = inst.params

        self._last_flow_id += 1
        if_id = self._last_flow_id
        if_name = f'if_{if_id}'

        ret = QuantumCircuit()
        for qr in true_body.qregs:
            ret.add_register(qr)
        for cr in true_body.cregs:
            ret.add_register(cr)
        if len(ret.clbits) == 0:
            ret.add_bits(true_body.clbits)
        if len(ret.qubits) == 0:
            ret.add_bits(true_body.qubits)

        if_true_label = f'{if_name}_true'
        if_else_label = f'{if_name}_else'
        if_end_label = f'{if_name}_end'

        ret.append(AerMark(if_true_label, ret.num_qubits),
                   range(ret.num_qubits), [])
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

        return condition_tuple, if_true_label, if_else_label if false_body else if_end_label, ret

def compile(circuits, backend):
    """
    compile a circuit that have control-flow instructions
    """
    return AerCompiler(backend).compile(circuits)
