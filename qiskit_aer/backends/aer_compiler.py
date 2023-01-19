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

import itertools

from qiskit.circuit import QuantumCircuit, Clbit
from qiskit.extensions import Initialize
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
                # Resolve initialize
                circuit = self._inline_initialize(circuit, compiled_optypes[idx])
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

    def _inline_initialize(self, circ, optype):
        """inline initialize.definition gates if statevector is not used"""
        if isinstance(optype, set) and Initialize not in optype:
            return circ

        for inst, _, _ in circ.data:
            if isinstance(inst, Initialize) and not isinstance(inst.params[0], complex):
                break
        else:
            return circ

        new_circ = circ.copy()
        new_circ.data = []
        for inst, qargs, cargs in circ.data:
            if isinstance(inst, Initialize) and not isinstance(inst.params[0], complex):
                # Assume that the decomposed circuit of inst.definition consists of basis gates
                new_circ.compose(inst.definition.decompose(), qargs, cargs, inplace=True)
            else:
                new_circ._append(inst, qargs, cargs)

        return new_circ

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
        for instruction in circuit.data:
            if isinstance(instruction.operation, controlflow_types):
                return True

        return False

    def _inline_circuit(self, circ, continue_label, break_label, bit_map=None):
        """convert control-flow instructions to mark and jump instructions

        Args:
            circ (QuantumCircuit): The QuantumCircuit to be compiled
            continue_label (str): label name for continue.
            break_label (str): label name for break.
            bit_map (dict[Bit, Bit]): mapping of virtual bits in the current circuit to the bit they
                represent in the outermost circuit.

        Returns:
            QuantumCircuit: QuantumCircuit without control-flow instructions
        """
        ret = circ.copy_empty_like()
        bit_map = {bit: bit for bit in itertools.chain(ret.qubits, ret.clbits)}

        for instruction in circ.data:
            # The barriers around all control-flow operations is to prevent any non-control-flow
            # operations from ending up topologically "inside" a body.  This can happen if the body
            # is not full width on the circuit, and the other operation uses disjoint bits.
            if isinstance(instruction.operation, ForLoopOp):
                ret.barrier()
                self._inline_for_loop_op(instruction, ret, bit_map)
                ret.barrier()
            elif isinstance(instruction.operation, WhileLoopOp):
                ret.barrier()
                self._inline_while_loop_op(instruction, ret, bit_map)
                ret.barrier()
            elif isinstance(instruction.operation, IfElseOp):
                ret.barrier()
                self._inline_if_else_op(instruction, continue_label, break_label, ret, bit_map)
                ret.barrier()
            elif isinstance(instruction.operation, BreakLoopOp):
                ret._append(
                    AerJump(break_label, ret.num_qubits, ret.num_clbits), ret.qubits, ret.clbits
                )
            elif isinstance(instruction.operation, ContinueLoopOp):
                ret._append(
                    AerJump(continue_label, ret.num_qubits, ret.num_clbits), ret.qubits, ret.clbits
                )
            else:
                ret._append(instruction)

        return ret

    def _convert_c_if_args(self, cond_tuple, bit_map):
        """Convert a condition tuple according to the wire map."""
        if isinstance(cond_tuple[0], Clbit):
            return (bit_map[cond_tuple[0]], cond_tuple[1])
        # ClassicalRegister conditions should already be in the outer circuit.
        return cond_tuple

    def _inline_for_loop_op(self, instruction, parent, bit_map):
        """inline for_loop body while iterating its indexset"""
        qargs = [bit_map[q] for q in instruction.qubits]
        cargs = [bit_map[c] for c in instruction.clbits]
        indexset, loop_parameter, body = instruction.operation.params
        inner_bit_map = {
            inner: bit_map[outer]
            for inner, outer in itertools.chain(
                zip(body.qubits, instruction.qubits),
                zip(body.clbits, instruction.clbits),
            )
        }

        self._last_flow_id += 1
        loop_id = self._last_flow_id
        loop_name = f'loop_{loop_id}'

        inlined_body = None
        break_label = f'{loop_name}_end'
        for index in indexset:
            continue_label = f'{loop_name}_{index}'
            inlined_body = self._inline_circuit(body, continue_label, break_label, inner_bit_map)
            if loop_parameter is not None:
                inlined_body = inlined_body.bind_parameters({loop_parameter: index})
            parent.append(inlined_body, qargs, cargs)
            parent.append(AerMark(continue_label, len(qargs), len(cargs)), qargs, cargs)

        if inlined_body is not None:
            parent.append(AerMark(break_label, len(qargs), len(cargs)), qargs, cargs)

    def _inline_while_loop_op(self, instruction, parent, bit_map):
        """inline while_loop body with jump and mark instructions"""
        condition_tuple = self._convert_c_if_args(instruction.operation.condition, bit_map)
        body, = instruction.operation.params

        self._last_flow_id += 1
        loop_id = self._last_flow_id
        loop_name = f'while_{loop_id}'

        continue_label = f'{loop_name}_continue'
        loop_start_label = f'{loop_name}_start'
        break_label = f'{loop_name}_end'
        inlined_body = self._inline_circuit(
            body,
            continue_label,
            break_label,
            {
                inner: bit_map[outer]
                for inner, outer in itertools.chain(
                    zip(body.qubits, instruction.qubits),
                    zip(body.clbits, instruction.clbits),
                )
            },
        )
        qargs = [bit_map[q] for q in instruction.qubits]
        cargs = [bit_map[c] for c in instruction.clbits]
        mark_cargs = cargs.copy()
        mark_cargs.extend(
            bit_map[c] for c in (
                (
                    {condition_tuple[0]} if isinstance(condition_tuple[0], Clbit)
                    else set(condition_tuple[0])
                ) - set(instruction.clbits)
            )
        )
        c_if_args = self._convert_c_if_args(condition_tuple, bit_map)

        parent.append(AerMark(continue_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)
        parent.append(AerJump(loop_start_label, len(qargs), len(mark_cargs)).c_if(*c_if_args),
                      qargs, mark_cargs)
        parent.append(AerJump(break_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)
        parent.append(AerMark(loop_start_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)
        parent.append(inlined_body, qargs, cargs)
        parent.append(AerJump(continue_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)
        parent.append(AerMark(break_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)

    def _inline_if_else_op(self, instruction, continue_label, break_label, parent, bit_map):
        """inline true and false bodies of if_else with jump and mark instructions"""
        condition_tuple = instruction.operation.condition
        true_body, false_body = instruction.operation.params

        self._last_flow_id += 1
        if_id = self._last_flow_id
        if_name = f'if_{if_id}'

        if_true_label = f'{if_name}_true'
        if_end_label = f'{if_name}_end'
        if false_body:
            if_else_label = f'{if_name}_else'
        else:
            if_else_label = if_end_label

        c_if_args = self._convert_c_if_args(condition_tuple, bit_map)

        qargs = [bit_map[q] for q in instruction.qubits]
        cargs = [bit_map[c] for c in instruction.clbits]
        mark_cargs = cargs.copy()
        mark_cargs.extend(
            bit_map[c] for c in (
                (
                    {condition_tuple[0]} if isinstance(condition_tuple[0], Clbit)
                    else set(condition_tuple[0])
                ) - set(instruction.clbits)
            )
        )

        true_bit_map = {
            inner: bit_map[outer]
            for inner, outer in itertools.chain(
                zip(true_body.qubits, instruction.qubits),
                zip(true_body.clbits, instruction.clbits),
            )
        }

        parent.append(
            AerJump(if_true_label, len(qargs), len(mark_cargs)).c_if(*c_if_args), qargs, mark_cargs
        )
        parent.append(AerJump(if_else_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)
        parent.append(AerMark(if_true_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)
        parent.append(
            self._inline_circuit(true_body, continue_label, break_label, true_bit_map), qargs, cargs
        )

        if false_body:
            false_bit_map = {
                inner: bit_map[outer]
                for inner, outer in itertools.chain(
                    zip(false_body.qubits, instruction.qubits),
                    zip(false_body.clbits, instruction.clbits),
                )
            }
            parent.append(AerJump(if_end_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)
            parent.append(AerMark(if_else_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)
            parent.append(
                self._inline_circuit(false_body, continue_label, break_label, false_bit_map),
                qargs,
                cargs,
            )

        parent.append(AerMark(if_end_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)


def compile_circuit(circuits, basis_gates=None, optypes=None):
    """
    compile a circuit that have control-flow instructions
    """
    return AerCompiler().compile(circuits, basis_gates, optypes)
