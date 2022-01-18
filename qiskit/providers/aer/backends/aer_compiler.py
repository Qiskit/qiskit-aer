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
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.circuit.controlflow import (
    WhileLoopOp,
    ForLoopOp,
    IfElseOp,
    BreakLoopOp,
    ContinueLoopOp)
from qiskit.compiler import transpile
from .backend_utils import circuit_optypes
from ..library.control_flow_instructions import AerMark, AerJump


class AerCompiler:
    """ Aer Compiler to convert instructions of control-flow to mark and jump instructions"""

    def __init__(self):
        self._last_flow_id = -1

    def compile(self, circuits, basis_gates=None, optypes=None):
        """compile a circuit that have control-flow instructions.

        Args:
            circuits (QuantumCircuit or list): The QuantumCircuits to be compiled
            basis_gates (list): basis gates to decompose sub-circuits
                                (default: None).
            optypes (list): list of instruction type sets for each circuit
                            (default: None).

        Returns:
            list: A list QuantumCircuit without control-flow
                  if optypes is None.
            tuple: A tuple of a list of quantum circuits and list of
                   compiled circuit optypes for each circuit if
                   optypes kwarg is not None.
        """
        if isinstance(circuits, (QuantumCircuit, Schedule, ScheduleBlock)):
            circuits = [circuits]
        if optypes is None:
            compiled_optypes = len(circuits) * [None]
        else:
            # Make a shallow copy incase we modify it
            compiled_optypes = list(optypes)
        if isinstance(circuits, list):
            basis_gates = basis_gates + ['mark', 'jump']
            compiled_circuits = []
            for idx, circuit in enumerate(circuits):
                if self._is_dynamic(circuit, compiled_optypes[idx]):
                    compiled_circ = transpile(
                        self._inline_circuit(circuit, None, None),
                        basis_gates=basis_gates
                    )
                    compiled_circuits.append(compiled_circ)
                    # Recompute optype for compiled circuit
                    compiled_optypes[idx] = circuit_optypes(compiled_circ)
                else:
                    compiled_circuits.append(circuit)
            if optypes is None:
                return compiled_circuits
            return compiled_circuits, compiled_optypes

        if optypes is None:
            return circuits
        return circuits, optypes

    @staticmethod
    def _is_dynamic(circuit, optype=None):
        """check whether a circuit contains control-flow instructions"""
        if not isinstance(circuit, QuantumCircuit):
            return False

        controlflow_types = (
            WhileLoopOp, ForLoopOp, IfElseOp, BreakLoopOp, ContinueLoopOp
        )

        # Check via optypes
        if isinstance(optype, set):
            return bool(optype.intersection(controlflow_types))

        # Check via iteration
        for inst, _, _ in circuit.data:
            if isinstance(inst, controlflow_types):
                return True

        return False

    def _inline_circuit(self, circ, continue_label, break_label):
        """convert control-flow instructions to mark and jump instructions

        Args:
            circ (QuantumCircuit): The QuantumCircuit to be compiled
            continue_label (str): label name for continue.
            break_label (str): label name for break.

        Returns:
            QuantumCircuit: QuantumCircuit without control-flow instructions
        """
        ret = circ.copy()
        ret.data = []

        q2i = {}
        for q in ret.qubits:
            q2i[q] = len(q2i)
        c2i = {}
        for c in ret.clbits:
            c2i[c] = len(c2i)

        for inst, qargs, cargs in circ.data:
            binding_qargs = [q2i[q] for q in qargs]
            binding_cargs = [c2i[c] for c in cargs]
            if isinstance(inst, ForLoopOp):
                self._inline_for_loop_op(inst, ret, binding_qargs, binding_cargs)
            elif isinstance(inst, WhileLoopOp):
                self._inline_while_loop_op(inst, ret, binding_qargs, binding_cargs)
            elif isinstance(inst, IfElseOp):
                self._inline_if_else_op(inst, continue_label, break_label,
                                        ret, binding_qargs, binding_cargs)
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
        """convert a boolean value to 0 or 1 in c_if elements"""
        return [1 if elem is True else 0 if elem is False else elem for elem in cond_tuple]

    def _inline_for_loop_op(self, inst, parent, qargs, cargs):
        """inline for_loop body while iterating its indexset"""
        indexset, loop_parameter, body = inst.params

        self._last_flow_id += 1
        loop_id = self._last_flow_id
        loop_name = f'loop_{loop_id}'

        inlined_body = None
        break_label = f'{loop_name}_end'
        for index in indexset:
            continue_label = f'{loop_name}_{index}'
            inlined_body = self._inline_circuit(body,
                                                continue_label,
                                                break_label)
            if loop_parameter is not None:
                inlined_body = inlined_body.bind_parameters({loop_parameter: index})
            parent.append(inlined_body, qargs, cargs)
            parent.append(AerMark(continue_label, inlined_body.num_qubits), qargs, [])

        if inlined_body:
            parent.append(AerMark(break_label, inlined_body.num_qubits), qargs, [])

    def _inline_while_loop_op(self, inst, parent, qargs, cargs):
        """inline while_loop body with jump and mark instructions"""
        condition_tuple = inst.condition
        body, = inst.params

        self._last_flow_id += 1
        loop_id = self._last_flow_id
        loop_name = f'while_{loop_id}'

        continue_label = f'{loop_name}_continue'
        loop_start_label = f'{loop_name}_start'
        break_label = f'{loop_name}_end'
        inlined_body = self._inline_circuit(body, continue_label, break_label)

        c_if_args = self._convert_c_if_args(condition_tuple)

        parent.append(AerMark(continue_label, inlined_body.num_qubits), qargs, [])
        parent.append(AerJump(loop_start_label, inlined_body.num_qubits).c_if(*c_if_args),
                      qargs, [])
        parent.append(AerJump(break_label, inlined_body.num_qubits), qargs, [])
        parent.append(AerMark(loop_start_label, inlined_body.num_qubits), qargs, [])
        parent.append(inlined_body, qargs, cargs)
        parent.append(AerJump(continue_label, inlined_body.num_qubits), qargs, [])
        parent.append(AerMark(break_label, inlined_body.num_qubits), qargs, [])

    def _inline_if_else_op(self, inst, continue_label, break_label, parent, qargs, cargs):
        """inline true and false bodies of if_else with jump and mark instructions"""
        condition_tuple = inst.condition
        true_body, false_body = inst.params

        self._last_flow_id += 1
        if_id = self._last_flow_id
        if_name = f'if_{if_id}'

        if_true_label = f'{if_name}_true'
        if_end_label = f'{if_name}_end'
        if false_body:
            if_else_label = f'{if_name}_else'
        else:
            if_else_label = if_end_label

        c_if_args = self._convert_c_if_args(condition_tuple)

        parent.append(AerJump(if_true_label, true_body.num_qubits).c_if(*c_if_args), qargs, [])
        parent.append(AerJump(if_else_label, true_body.num_qubits), qargs, [])
        parent.append(AerMark(if_true_label, true_body.num_qubits), qargs, [])
        parent.append(self._inline_circuit(true_body, continue_label, break_label), qargs, cargs)

        if false_body:
            parent.append(AerJump(if_end_label, true_body.num_qubits), qargs, [])
            parent.append(AerMark(if_else_label, true_body.num_qubits), qargs, [])
            parent.append(self._inline_circuit(false_body, continue_label, break_label),
                          qargs, cargs)

        parent.append(AerMark(if_end_label, true_body.num_qubits), qargs, [])


def compile_circuit(circuits, basis_gates=None, optypes=None):
    """
    compile a circuit that have control-flow instructions
    """
    return AerCompiler().compile(circuits, basis_gates, optypes)
