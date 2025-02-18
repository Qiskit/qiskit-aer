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

import collections
import itertools
from copy import copy
from typing import List
from warnings import warn
from concurrent.futures import Executor
import uuid
import numpy as np

from qiskit.circuit import QuantumCircuit, Clbit, ClassicalRegister, ParameterExpression
from qiskit.circuit.classical.expr import Expr, Unary, Binary, Var, Value, ExprVisitor, iter_vars
from qiskit.circuit.classical.types import Bool, Uint
from qiskit.circuit.library import Initialize
from qiskit.providers.options import Options
from qiskit.circuit import Store
from qiskit.circuit.controlflow import (
    WhileLoopOp,
    ForLoopOp,
    IfElseOp,
    BreakLoopOp,
    ContinueLoopOp,
    SwitchCaseOp,
    CASE_DEFAULT,
)
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Decompose

from qiskit_aer.aererror import AerError
from qiskit_aer.noise import NoiseModel

# pylint: disable=import-error, no-name-in-module
from qiskit_aer.backends.controller_wrappers import (
    AerUnaryExpr,
    AerUnaryOp,
    AerBinaryExpr,
    AerBinaryOp,
    AerUintValue,
    AerBoolValue,
    AerUint,
    AerBool,
    AerCast,
    AerVar,
    AerCircuit,
    AerConfig,
)

from .backend_utils import circuit_optypes, CircuitHeader
from ..library.control_flow_instructions import AerMark, AerJump, AerStore


class AerCompiler:
    """Aer Compiler to convert instructions of control-flow to mark and jump instructions"""

    def __init__(self):
        self._last_flow_id = -1

    def compile(self, circuits, optypes=None):
        """compile a circuit that have control-flow instructions.

        Args:
            circuits (QuantumCircuit or list): The QuantumCircuits to be compiled
            optypes (list): list of instruction type sets for each circuit
                            (default: None).

        Returns:
            list: A list QuantumCircuit without control-flow
                  if optypes is None.
            tuple: A tuple of a list of quantum circuits and list of
                   compiled circuit optypes for each circuit if
                   optypes kwarg is not None.
        """
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        if optypes is None:
            compiled_optypes = len(circuits) * [None]
        else:
            # Make a shallow copy incase we modify it
            compiled_optypes = list(optypes)
        if isinstance(circuits, list):
            compiled_circuits = []
            for idx, circuit in enumerate(circuits):
                # Resolve initialize
                circuit = self._inline_initialize(circuit, compiled_optypes[idx])
                if self._is_dynamic(circuit, compiled_optypes[idx]):
                    pm = PassManager([Decompose(["mark", "jump"])])
                    compiled_circ = pm.run(self._inline_circuit(circuit, None, None))
                    # compiled_circ._vars_local = inlined_circ._vars_local
                    # compiled_circ._vars_input = inlined_circ._vars_input
                    # compiled_circ._vars_capture = inlined_circ._vars_capture
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

        for datum in circ.data:
            inst = datum.operation
            if isinstance(inst, Initialize) and (
                (not isinstance(inst.params[0], complex)) or (len(inst.params) == 1)
            ):
                break
        else:
            return circ

        new_circ = circ.copy()
        new_circ.data = []
        for datum in circ.data:
            inst, qargs, cargs = datum.operation, datum.qubits, datum.clbits
            if isinstance(inst, Initialize) and (
                (not isinstance(inst.params[0], complex)) or (len(inst.params) == 1)
            ):
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
            WhileLoopOp,
            ForLoopOp,
            IfElseOp,
            BreakLoopOp,
            ContinueLoopOp,
            SwitchCaseOp,
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
            elif isinstance(instruction.operation, SwitchCaseOp):
                ret.barrier()
                self._inline_switch_case_op(instruction, continue_label, break_label, ret, bit_map)
                ret.barrier()
            elif isinstance(instruction.operation, BreakLoopOp):
                ret._append(
                    AerJump(break_label, ret.num_qubits, ret.num_clbits), ret.qubits, ret.clbits
                )
            elif isinstance(instruction.operation, ContinueLoopOp):
                ret._append(
                    AerJump(continue_label, ret.num_qubits, ret.num_clbits), ret.qubits, ret.clbits
                )
            elif isinstance(instruction.operation, Store):
                ret._append(
                    AerStore(ret.num_qubits, ret.num_clbits, instruction.operation),
                    ret.qubits,
                    ret.clbits,
                )
            else:
                ret._append(instruction)
        return ret

    def _convert_jump_conditional(self, cond_tuple, bit_map):
        """Convert a condition tuple according to the wire map."""
        if isinstance(cond_tuple, Expr):
            return cond_tuple
        elif isinstance(cond_tuple[0], Clbit):
            return (bit_map[cond_tuple[0]], cond_tuple[1])
        elif isinstance(cond_tuple[0], ClassicalRegister):
            # ClassicalRegister conditions should already be in the outer circuit.
            return cond_tuple
        elif isinstance(cond_tuple[0], Var):
            if isinstance(cond_tuple[0].var, Clbit):
                expr = Var(bit_map[cond_tuple[0].var], cond_tuple[0].type)
            elif isinstance(cond_tuple[0].var, ClassicalRegister):
                expr = Var([bit_map[clbit] for clbit in cond_tuple[0].var], cond_tuple[0].type)
            else:
                raise AerError(
                    f"jump condition does not support this tyep of Var: {cond_tuple[0]}."
                )
            return (expr, cond_tuple[1])

        raise AerError(f"jump condition does not support {cond_tuple[0].__class__}.")

    def _list_clbit_from_expr(self, bit_map, expr):
        ret = set()
        for var in iter_vars(expr):
            if isinstance(var.var, Clbit):
                ret.add(bit_map[var.var])
            elif isinstance(var.var, ClassicalRegister):
                ret.update(bit_map[bit] for bit in var.var)
        return ret

    def _inline_for_loop_op(self, instruction, parent, bit_map):
        """inline for_loop body while iterating its indexset"""
        qargs = [bit_map[q] for q in instruction.qubits]
        cargs = [bit_map[c] for c in instruction.clbits]
        # to avoid wrong topological sorting of command with "empty" block
        if len(qargs) == 0:
            qargs = parent.qubits
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
        loop_name = f"loop_{loop_id}"

        inlined_body = None
        break_label = f"{loop_name}_end"
        for index in indexset:
            continue_label = f"{loop_name}_{index}"
            inlined_body = self._inline_circuit(body, continue_label, break_label, inner_bit_map)
            if loop_parameter is not None:
                inlined_body = inlined_body.assign_parameters({loop_parameter: index})
            for inst in inlined_body:
                parent.append(
                    inst.replace(
                        qubits=[inner_bit_map[bit] for bit in inst.qubits],
                        clbits=[inner_bit_map[bit] for bit in inst.clbits],
                    ),
                    qargs,
                    cargs,
                )
            parent.append(AerMark(continue_label, len(qargs), len(cargs)), qargs, cargs)

        if inlined_body is not None:
            parent.append(AerMark(break_label, len(qargs), len(cargs)), qargs, cargs)

    def _inline_while_loop_op(self, instruction, parent, bit_map):
        """inline while_loop body with jump and mark instructions"""
        condition_tuple = self._convert_jump_conditional(instruction.operation.condition, bit_map)
        (body,) = instruction.operation.params

        self._last_flow_id += 1
        loop_id = self._last_flow_id
        loop_name = f"while_{loop_id}"

        continue_label = f"{loop_name}_continue"
        loop_start_label = f"{loop_name}_start"
        break_label = f"{loop_name}_end"
        inline_bit_map = {
            inner: bit_map[outer]
            for inner, outer in itertools.chain(
                zip(body.qubits, instruction.qubits),
                zip(body.clbits, instruction.clbits),
            )
        }
        inlined_body = self._inline_circuit(
            body,
            continue_label,
            break_label,
            inline_bit_map,
        )
        qargs = [bit_map[q] for q in instruction.qubits]
        cargs = [bit_map[c] for c in instruction.clbits]
        # to avoid wrong topological sorting of command with "empty" block
        if len(qargs) == 0:
            qargs = parent.qubits

        if isinstance(condition_tuple, Expr):
            mark_cargs = self._list_clbit_from_expr(bit_map, condition_tuple)
        elif isinstance(condition_tuple[0], Clbit):
            mark_cargs = {bit_map[condition_tuple[0]]}
        else:
            mark_cargs = {bit_map[c] for c in condition_tuple[0]}
        mark_cargs = set(cargs).union(mark_cargs) - set(instruction.clbits)

        c_if_args = self._convert_jump_conditional(condition_tuple, bit_map)

        parent.append(AerMark(continue_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)
        parent.append(
            AerJump(loop_start_label, len(qargs), len(mark_cargs)).set_conditional(c_if_args),
            qargs,
            mark_cargs,
        )
        parent.append(AerJump(break_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)
        parent.append(AerMark(loop_start_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)
        for inst in inlined_body:
            parent.append(
                inst.replace(
                    qubits=[inline_bit_map[bit] for bit in inst.qubits],
                    clbits=[inline_bit_map[bit] for bit in inst.clbits],
                ),
                qargs,
                cargs,
            )
        parent.append(AerJump(continue_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)
        parent.append(AerMark(break_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)

    def _inline_if_else_op(self, instruction, continue_label, break_label, parent, bit_map):
        """inline true and false bodies of if_else with jump and mark instructions"""
        condition_tuple = instruction.operation.condition
        true_body, false_body = instruction.operation.params

        self._last_flow_id += 1
        if_id = self._last_flow_id
        if_name = f"if_{if_id}"

        if_true_label = f"{if_name}_true"
        if_end_label = f"{if_name}_end"
        if false_body:
            if_else_label = f"{if_name}_else"
        else:
            if_else_label = if_end_label

        c_if_args = self._convert_jump_conditional(condition_tuple, bit_map)

        qargs = [bit_map[q] for q in instruction.qubits]
        cargs = [bit_map[c] for c in instruction.clbits]
        # to avoid wrong topological sorting of command with "empty" block
        if len(qargs) == 0:
            qargs = parent.qubits

        # to avoid wrong topological sorting of command with "empty" block
        if len(qargs) == 0:
            qargs = parent.qubits

        if isinstance(condition_tuple, Expr):
            mark_cargs = self._list_clbit_from_expr(bit_map, condition_tuple)
        elif isinstance(condition_tuple[0], Clbit):
            mark_cargs = {bit_map[condition_tuple[0]]}
        else:
            mark_cargs = {bit_map[c] for c in condition_tuple[0]}
        mark_cargs = set(cargs).union(mark_cargs) - set(instruction.clbits)

        true_bit_map = {
            inner: bit_map[outer]
            for inner, outer in itertools.chain(
                zip(true_body.qubits, instruction.qubits),
                zip(true_body.clbits, instruction.clbits),
            )
        }

        parent.append(
            AerJump(if_true_label, len(qargs), len(mark_cargs)).set_conditional(c_if_args),
            qargs,
            mark_cargs,
        )
        parent.append(AerJump(if_else_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)
        parent.append(AerMark(if_true_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)
        child = self._inline_circuit(true_body, continue_label, break_label, true_bit_map)
        for inst in child.data:
            parent.append(
                inst.replace(
                    qubits=[true_bit_map[bit] for bit in inst.qubits],
                    clbits=[true_bit_map[bit] for bit in inst.clbits],
                ),
                qargs,
                cargs,
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
            child = self._inline_circuit(false_body, continue_label, break_label, false_bit_map)
            for inst in child.data:
                parent.append(
                    inst.replace(
                        qubits=[false_bit_map[bit] for bit in inst.qubits],
                        clbits=[false_bit_map[bit] for bit in inst.clbits],
                    ),
                    qargs,
                    cargs,
                )

        parent.append(AerMark(if_end_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)

    def _inline_switch_case_op(self, instruction, continue_label, break_label, parent, bit_map):
        """inline switch cases with jump and mark instructions"""
        cases = instruction.operation.cases_specifier()

        self._last_flow_id += 1
        switch_id = self._last_flow_id
        switch_name = f"switch_{switch_id}"

        qargs = [bit_map[q] for q in instruction.qubits]
        cargs = [bit_map[c] for c in instruction.clbits]
        # to avoid wrong topological sorting of command with "empty" block
        if len(qargs) == 0:
            qargs = parent.qubits

        if isinstance(instruction.operation.target, Clbit):
            target_clbits = {bit_map[instruction.operation.target]}
        elif isinstance(instruction.operation.target, Expr):
            target_clbits = self._list_clbit_from_expr(bit_map, instruction.operation.target)
        else:
            target_clbits = {bit_map[c] for c in instruction.operation.target}
        mark_cargs = set(cargs).union(target_clbits) - set(instruction.clbits)

        switch_end_label = f"{switch_name}_end"
        case_default_label = None
        CaseData = collections.namedtuple("CaseData", ["label", "args_list", "bit_map", "body"])
        case_data_list = []
        for i, case in enumerate(cases):
            if case_default_label is not None:
                raise AerError("cases after the default are unreachable")

            case_data = CaseData(
                label=f"{switch_name}_{i}",
                args_list=[
                    (
                        self._convert_jump_conditional(
                            (instruction.operation.target, switch_val), bit_map
                        )
                        if switch_val != CASE_DEFAULT
                        else []
                    )
                    for switch_val in case[0]
                ],
                bit_map={
                    inner: bit_map[outer]
                    for inner, outer in itertools.chain(
                        zip(case[1].qubits, instruction.qubits),
                        zip(case[1].clbits, instruction.clbits),
                    )
                },
                body=case[1],
            )
            case_data_list.append(case_data)
            if CASE_DEFAULT in case[0]:
                case_default_label = case_data.label

        if case_default_label is None:
            case_default_label = switch_end_label

        for case_data in case_data_list:
            for case_args in case_data.args_list:
                if len(case_args) > 0:
                    if isinstance(case_args[0], Expr):
                        case_args = Binary(
                            Binary.Op.EQUAL,
                            case_args[0],
                            Value(case_args[1], case_args[0].type),
                            Bool(),
                        )
                    parent.append(
                        AerJump(case_data.label, len(qargs), len(mark_cargs)).set_conditional(
                            case_args
                        ),
                        qargs,
                        mark_cargs,
                    )

        parent.append(AerJump(case_default_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)

        for case_data in case_data_list:
            parent.append(AerMark(case_data.label, len(qargs), len(mark_cargs)), qargs, mark_cargs)
            child = self._inline_circuit(
                case_data.body, continue_label, break_label, case_data.bit_map
            )
            for inst in child.data:
                parent.append(
                    inst.replace(
                        qubits=[case_data.bit_map[bit] for bit in inst.qubits],
                        clbits=[case_data.bit_map[bit] for bit in inst.clbits],
                    ),
                    qargs,
                    cargs,
                )
            parent.append(AerJump(switch_end_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)

        parent.append(AerMark(switch_end_label, len(qargs), len(mark_cargs)), qargs, mark_cargs)


def compile_circuit(circuits, optypes=None):
    """
    compile a circuit that have control-flow instructions
    """
    return AerCompiler().compile(circuits, optypes)


BACKEND_RUN_ARG_TYPES = {
    "shots": (int, np.integer),
    "method": (str),
    "device": (str),
    "precision": (str),
    "max_job_size": (int, np.integer),
    "max_shot_size": (int, np.integer),
    "enable_truncation": (bool, np.bool_),
    "executor": Executor,
    "zero_threshold": (float, np.floating),
    "validation_threshold": (int, np.integer),
    "max_parallel_threads": (int, np.integer),
    "max_parallel_experiments": (int, np.integer),
    "max_parallel_shots": (int, np.integer),
    "max_memory_mb": (int, np.integer),
    "fusion_enable": (bool, np.bool_),
    "fusion_verbose": (bool, np.bool_),
    "fusion_max_qubit": (int, np.integer),
    "fusion_threshold": (int, np.integer),
    "accept_distributed_results": (bool, np.bool_),
    "memory": (bool, np.bool_),
    "noise_model": (NoiseModel),
    "seed_simulator": (int, np.integer),
    "cuStateVec_enable": (int, np.integer),
    "blocking_qubits": (int, np.integer),
    "blocking_enable": (bool, np.bool_),
    "chunk_swap_buffer_qubits": (int, np.integer),
    "batched_shots_gpu": (bool, np.bool_),
    "batched_shots_gpu_max_qubits": (int, np.integer),
    "shot_branching_enable": (bool, np.bool_),
    "shot_branching_sampling_enable": (bool, np.bool_),
    "num_threads_per_device": (int, np.integer),
    "statevector_parallel_threshold": (int, np.integer),
    "statevector_sample_measure_opt": (int, np.integer),
    "stabilizer_max_snapshot_probabilities": (int, np.integer),
    "extended_stabilizer_sampling_method": (str),
    "extended_stabilizer_metropolis_mixing_time": (int, np.integer),
    "extended_stabilizer_approximation_error": (float, np.floating),
    "extended_stabilizer_norm_estimation_samples": (int, np.integer),
    "extended_stabilizer_norm_estimation_repetitions": (int, np.integer),
    "extended_stabilizer_parallel_threshold": (int, np.integer),
    "extended_stabilizer_probabilities_snapshot_samples": (int, np.integer),
    "matrix_product_state_truncation_threshold": (float, np.floating),
    "matrix_product_state_max_bond_dimension": (int, np.integer),
    "mps_sample_measure_algorithm": (str),
    "mps_log_data": (bool, np.bool_),
    "mps_swap_direction": (str),
    "chop_threshold": (float, np.floating),
    "mps_parallel_threshold": (int, np.integer),
    "mps_omp_threads": (int, np.integer),
    "mps_lapack": (bool, np.bool_),
    "tensor_network_num_sampling_qubits": (int, np.integer),
    "use_cuTensorNet_autotuning": (bool, np.bool_),
    "parameterizations": (list),
    "fusion_parallelization_threshold": (int, np.integer),
    "target_gpus": (list),
    "runtime_parameter_bind_enable": (bool, np.bool_),
}


def _validate_option(k, v):
    """validate backend.run arguments"""
    if v is None:
        return v
    if k not in BACKEND_RUN_ARG_TYPES:
        raise AerError(f"invalid argument: name={k}")
    if isinstance(v, BACKEND_RUN_ARG_TYPES[k]):
        return v

    expected_type = BACKEND_RUN_ARG_TYPES[k][0]

    if expected_type in (int, float, bool, str):
        try:
            ret = expected_type(v)
            if not isinstance(v, BACKEND_RUN_ARG_TYPES[k]):
                warn(
                    f'A type of an option "{k}" should be {expected_type.__name__} '
                    "but {v.__class__.__name__} was specified."
                    "Implicit cast for an argument has been deprecated as of qiskit-aer 0.12.1.",
                    DeprecationWarning,
                    stacklevel=5,
                )
            return ret
        except Exception:  # pylint: disable=broad-except
            pass

    raise TypeError(
        f"invalid option type: name={k}, "
        f"type={v.__class__.__name__}, expected={BACKEND_RUN_ARG_TYPES[k][0].__name__}"
    )


def generate_aer_config(
    circuits: List[QuantumCircuit], backend_options: Options, **run_options
) -> AerConfig:
    """generates a configuration to run simulation.

    Args:
        circuits: circuit(s) to be converted
        backend_options: backend options
        run_options: run options

    Returns:
        AerConfig to run Aer
    """
    num_qubits = max(circuit.num_qubits for circuit in circuits)
    memory_slots = max(circuit.num_clbits for circuit in circuits)

    config = AerConfig()
    config.memory_slots = memory_slots
    config.n_qubits = num_qubits
    for key, value in backend_options.__dict__.items():
        if hasattr(config, key) and value is not None:
            value = _validate_option(key, value)
            setattr(config, key, value)
    for key, value in run_options.items():
        if hasattr(config, key) and value is not None:
            value = _validate_option(key, value)
            setattr(config, key, value)
    return config


def assemble_circuit(circuit: QuantumCircuit, basis_gates=None):
    """assemble circuit object mapped to AER::Circuit"""

    num_qubits = circuit.num_qubits
    num_memory = circuit.num_clbits
    extra_creg_idx = 0

    qreg_sizes = []
    creg_sizes = []
    if (
        isinstance(circuit.global_phase, ParameterExpression)
        and len(circuit.global_phase.parameters) > 0
    ):
        global_phase = 0.0
    else:
        global_phase = float(circuit.global_phase)

    for qreg in circuit.qregs:
        qreg_sizes.append([qreg.name, qreg.size])
    for creg in circuit.cregs:
        creg_sizes.append([creg.name, creg.size])

    is_conditional = any(
        getattr(inst.operation, "condition_expr", None)
        or getattr(inst.operation, "condition", None)
        for inst in circuit.data
    )

    header = CircuitHeader(
        n_qubits=num_qubits,
        qreg_sizes=qreg_sizes,
        memory_slots=num_memory,
        creg_sizes=creg_sizes,
        name=circuit.name,
        global_phase=global_phase,
    )

    qubit_indices = {qubit: idx for idx, qubit in enumerate(circuit.qubits)}
    clbit_indices = {clbit: idx for idx, clbit in enumerate(circuit.clbits)}

    aer_circ = AerCircuit()
    aer_circ.set_header(header)
    aer_circ.num_qubits = num_qubits
    aer_circ.num_memory = num_memory
    aer_circ.global_phase_angle = global_phase

    var_heap_map = {}
    for var in _iter_var_recursive(circuit):
        memory_pos = num_memory + extra_creg_idx
        var_heap_map[var.name] = (memory_pos, var.type.width)
        extra_creg_idx += var.type.width

    num_of_aer_ops = 0
    index_map = []
    for inst in circuit.data:
        # To convert to a qobj-style conditional, insert a bfunc prior
        # to the conditional instruction to map the creg ?= val condition
        # onto a gating register bit.
        conditional_reg = -1
        conditional_expr = None
        if hasattr(inst.operation, "condition") and inst.operation.condition:
            ctrl_reg, ctrl_val = inst.operation.condition
            mask = 0
            val = 0
            if isinstance(ctrl_reg, Clbit):
                mask = 1 << clbit_indices[ctrl_reg]
                val = (ctrl_val & 1) << clbit_indices[ctrl_reg]
            else:
                for clbit, idx in clbit_indices.items():
                    if clbit in ctrl_reg:
                        mask |= 1 << idx
                        val |= ((ctrl_val >> list(ctrl_reg).index(clbit)) & 1) << idx
            conditional_reg = num_memory + extra_creg_idx
            aer_circ.bfunc(f"0x{mask:X}", f"0x{val:X}", "==", conditional_reg)
            num_of_aer_ops += 1
            extra_creg_idx += 1
        elif hasattr(inst.operation, "condition_expr") and inst.operation.condition_expr:
            conditional_expr = inst.operation.condition_expr

        num_of_aer_ops += _assemble_op(
            circuit,
            aer_circ,
            inst,
            qubit_indices,
            clbit_indices,
            is_conditional,
            conditional_reg,
            conditional_expr,
            basis_gates,
        )
        index_map.append(num_of_aer_ops - 1)

    return aer_circ, index_map


def _assemble_type(expr_type):
    if isinstance(expr_type, Uint):
        return AerUint(expr_type.width)
    elif isinstance(expr_type, Bool):
        return AerBool()
    else:
        raise AerError(f"unknown type: {expr_type.__class__}")


def _iter_var_recursive(circuit):
    yield from circuit.iter_vars()
    for instruction in circuit.data:
        for param in instruction.operation.params:
            if isinstance(param, QuantumCircuit):
                yield from _iter_var_recursive(param)


def _find_var_clbits(circuit, var_uuid):
    clbit_index = circuit.num_clbits
    for var in _iter_var_recursive(circuit):
        if var.var == var_uuid:
            return list(range(clbit_index, clbit_index + var.type.width))
        clbit_index += var.type.width
    raise AerError(f"Var is not registed in this circuit: uuid={var_uuid}")


def _assemble_clbit_indices(circ, c):
    if isinstance(c, (ClassicalRegister, list)):
        return [circ.find_bit(cbit).index for cbit in c]
    elif isinstance(c, Clbit):
        return [circ.find_bit(c).index]
    elif isinstance(c, uuid.UUID):
        return _find_var_clbits(circ, c)
    else:
        raise AerError(f"unknown clibt list: {c.__class__}")


def _assemble_unary_operator(op):
    if op is Unary.Op.BIT_NOT:
        return AerUnaryOp.BitNot
    elif op is Unary.Op.LOGIC_NOT:
        return AerUnaryOp.LogicNot
    else:
        raise AerError(f"unknown op: {op}")


_BINARY_OP_MAPPING = {
    Binary.Op.BIT_AND: AerBinaryOp.BitAnd,
    Binary.Op.BIT_OR: AerBinaryOp.BitOr,
    Binary.Op.BIT_XOR: AerBinaryOp.BitXor,
    Binary.Op.LOGIC_AND: AerBinaryOp.LogicAnd,
    Binary.Op.LOGIC_OR: AerBinaryOp.LogicOr,
    Binary.Op.EQUAL: AerBinaryOp.Equal,
    Binary.Op.NOT_EQUAL: AerBinaryOp.NotEqual,
    Binary.Op.LESS: AerBinaryOp.Less,
    Binary.Op.LESS_EQUAL: AerBinaryOp.LessEqual,
    Binary.Op.GREATER: AerBinaryOp.Greater,
    Binary.Op.GREATER_EQUAL: AerBinaryOp.GreaterEqual,
}


def _assemble_binary_operator(op):
    if op in _BINARY_OP_MAPPING:
        return _BINARY_OP_MAPPING[op]
    else:
        raise AerError(f"unknown op: {op}")


class _AssembleExprImpl(ExprVisitor):
    """Convert from Expr objects to corresponding objects."""

    def __init__(self, circuit):
        self.circuit = circuit

    def visit_value(self, node, /):
        """return Aer's value types."""
        # pylint: disable=unused-variable
        if isinstance(node.type, Uint):
            return AerUintValue(node.type.width, node.value)
        elif isinstance(node.type, Bool):
            return AerBoolValue(node.value)
        else:
            raise AerError(f"invalid value type is specified: {node.type.__class__}")

    def visit_var(self, node, /):
        return AerVar(_assemble_type(node.type), _assemble_clbit_indices(self.circuit, node.var))

    def visit_cast(self, node, /):
        return AerCast(_assemble_type(node.type), node.operand.accept(self))

    def visit_unary(self, node, /):
        return AerUnaryExpr(_assemble_unary_operator(node.op), node.operand.accept(self))

    def visit_binary(self, node, /):
        return AerBinaryExpr(
            _assemble_binary_operator(node.op),
            node.left.accept(self),
            node.right.accept(self),
        )

    def visit_generic(self, node, /):
        raise AerError(f"unsupported expression is used: {node.__class__}")


def _check_no_conditional(inst_name, conditional_reg):
    if conditional_reg >= 0:
        raise AerError(f"instruction {inst_name} does not support conditional")


def _assemble_op(
    circ,
    aer_circ,
    inst,
    qubit_indices,
    clbit_indices,
    is_conditional,
    conditional_reg,
    conditional_expr,
    basis_gates,
):
    operation = inst.operation
    qubits = [qubit_indices[qubit] for qubit in inst.qubits]
    clbits = [clbit_indices[clbit] for clbit in inst.clbits]
    name = operation.name
    label = operation.label
    params = operation.params if hasattr(operation, "params") else None
    copied = False

    for i, param in enumerate(params):
        if isinstance(param, ParameterExpression) and len(param.parameters) > 0:
            if not copied:
                params = copy(params)
                copied = True
            params[i] = 0.0

    aer_cond_expr = conditional_expr.accept(_AssembleExprImpl(circ)) if conditional_expr else None

    # check if there is ctrl_state option
    ctrl_state_pos = name.find("_o")
    if ctrl_state_pos > 0:
        gate_name = name[0:ctrl_state_pos]
    else:
        gate_name = name

    num_of_aer_ops = 1
    # fmt: off
    if (gate_name in {
        "ccx", "ccz", "cp", "cswap", "csx", "cx", "cy", "cz", "delay", "ecr", "h",
        "id", "mcp", "mcphase", "mcr", "mcrx", "mcry", "mcrz", "mcswap", "mcsx",
        "mcu", "mcu1", "mcu2", "mcu3", "mcx", "mcx_gray", "mcy", "mcz", "p", "r",
        "rx", "rxx", "ry", "ryy", "rz", "rzx", "rzz", "s", "sdg", "swap", "sx", "sxdg",
        "t", "tdg", "u", "x", "y", "z", "u1", "u2", "u3", "cu", "cu1", "cu2", "cu3",
        "crx", "cry", "crz",
    }) and (basis_gates is None or gate_name in basis_gates):
        if ctrl_state_pos > 0:
            # Add x gates for ctrl qubits which state=0
            ctrl_state = int(name[ctrl_state_pos+2:len(name)])
            for i in range(len(qubits)-1):
                if (ctrl_state >> i) & 1 == 0:
                    qubits_i = [qubits[i]]
                    aer_circ.gate("x", qubits_i, params, [], conditional_reg, aer_cond_expr,
                                  label if label else "x")
                    num_of_aer_ops += 1
            aer_circ.gate(gate_name, qubits, params, [], conditional_reg, aer_cond_expr,
                          label if label else gate_name)
            for i in range(len(qubits)-1):
                if (ctrl_state >> i) & 1 == 0:
                    qubits_i = [qubits[i]]
                    aer_circ.gate("x", qubits_i, params, [], conditional_reg, aer_cond_expr,
                                  label if label else "x")
                    num_of_aer_ops += 1
        else:
            aer_circ.gate(name, qubits, params, [], conditional_reg, aer_cond_expr,
                          label if label else name)
    elif name == "measure":
        if is_conditional:
            aer_circ.measure(qubits, clbits, clbits)
        else:
            aer_circ.measure(qubits, clbits, [])
    elif name == "reset":
        aer_circ.reset(qubits, conditional_reg)
    elif name == "diagonal":
        aer_circ.diagonal(qubits, params, conditional_reg, label if label else "diagonal")
    elif name == "unitary":
        aer_circ.unitary(qubits, params[0], conditional_reg, aer_cond_expr,
                         label if label else "unitary")
    elif name == "pauli":
        aer_circ.gate(name, qubits, [], params, conditional_reg, aer_cond_expr,
                      label if label else name)
    elif name == "initialize":
        _check_no_conditional(name, conditional_reg)
        aer_circ.initialize(qubits, params)
    elif name == "roerror":
        _check_no_conditional(name, conditional_reg)
        aer_circ.roerror(qubits, params)
    elif name == "multiplexer":
        aer_circ.multiplexer(qubits, params, conditional_reg, aer_cond_expr, label if label else name)
    elif name == "kraus":
        aer_circ.kraus(qubits, params, conditional_reg, aer_cond_expr)
    elif name in {
        "save_statevector",
        "save_statevector_dict",
        "save_clifford",
        "save_probabilities",
        "save_probabilities_dict",
        "save_matrix_product_state",
        "save_unitary",
        "save_superop",
        "save_density_matrix",
        "save_state",
        "save_stabilizer",
    }:
        _check_no_conditional(name, conditional_reg)
        aer_circ.save_state(qubits, name, operation._subtype, label if label else name)
    elif name in {"save_amplitudes", "save_amplitudes_sq"}:
        _check_no_conditional(name, conditional_reg)
        aer_circ.save_amplitudes(qubits, name, params, operation._subtype, label if label else name)
    elif name in ("save_expval", "save_expval_var"):
        _check_no_conditional(name, conditional_reg)
        paulis = []
        coeff_reals = []
        coeff_imags = []
        for pauli, coeff in operation.params:
            paulis.append(pauli)
            coeff_reals.append(coeff[0])
            coeff_imags.append(coeff[1])
        aer_circ.save_expval(
            qubits,
            name,
            paulis,
            coeff_reals,
            coeff_imags,
            operation._subtype,
            label if label else name,
        )
    elif name == "set_statevector":
        _check_no_conditional(name, conditional_reg)
        aer_circ.set_statevector(qubits, params)
    elif name == "set_unitary":
        _check_no_conditional(name, conditional_reg)
        aer_circ.set_unitary(qubits, params)
    elif name == "set_density_matrix":
        _check_no_conditional(name, conditional_reg)
        aer_circ.set_density_matrix(qubits, params)
    elif name == "set_stabilizer":
        _check_no_conditional(name, conditional_reg)
        aer_circ.set_clifford(qubits, params)
    elif name == "set_superop":
        _check_no_conditional(name, conditional_reg)
        aer_circ.set_superop(qubits, params)
    elif name == "set_matrix_product_state":
        _check_no_conditional(name, conditional_reg)
        aer_circ.set_matrix_product_state(qubits, params)
    elif name == "superop":
        aer_circ.superop(qubits, params[0], conditional_reg, aer_cond_expr)
    elif name == "barrier":
        _check_no_conditional(name, conditional_reg)
        num_of_aer_ops = 0
    elif name == "jump":
        aer_circ.jump(qubits, params, conditional_reg, aer_cond_expr)
    elif name == "mark":
        _check_no_conditional(name, conditional_reg)
        aer_circ.mark(qubits, params)
    elif name == "qerror_loc":
        aer_circ.set_qerror_loc(qubits, label if label else name, conditional_reg, aer_cond_expr)
    elif name in ("for_loop", "while_loop", "if_else"):
        raise AerError(
            "control-flow instructions must be converted " f"to jump and mark instructions: {name}"
        )
    elif name == "aer_store":
        if not isinstance(operation.store.lvalue, Var):
            raise AerError(f"unsupported lvalue : {operation.store.lvalue.__class__}")
        aer_circ.store(qubits, _assemble_clbit_indices(circ, operation.store.lvalue.var),
                       operation.store.rvalue.accept(_AssembleExprImpl(circ)))
        num_of_aer_ops = 1
    elif name == "store":
        if not isinstance(operation.lvalue, Var):
            raise AerError(f"unsupported lvalue : {operation.lvalue.__class__}")
        aer_circ.store(qubits, _assemble_clbit_indices(circ, operation.lvalue.var),
                       operation.rvalue.accept(_AssembleExprImpl(circ)))
        num_of_aer_ops = 1
    else:
        raise AerError(f"unknown instruction: {name}")

    return num_of_aer_ops


def assemble_circuits(circuits: List[QuantumCircuit], basis_gates: list = None) -> List[AerCircuit]:
    """converts a list of Qiskit circuits into circuits mapped AER::Circuit

    Args:
        circuits: circuit(s) to be converted
        basis_gates (list): supported gates to be converted

    Returns:
        a list of circuits to be run on the Aer backends and
        a list of index mapping from Qiskit instructions to Aer operations of the circuits

    Examples:

        .. code-block:: python

            from qiskit.circuit import QuantumCircuit
            from qiskit_aer.backends.aer_compiler import assemble_circuits
            # Create a circuit to be simulated
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            # Generate AerCircuit from the input circuit
            aer_qc_list, idx_maps = assemble_circuits(circuits=[qc])
    """
    if basis_gates is not None:
        basis_gates_set = set(basis_gates)
        aer_circuits, idx_maps = zip(
            *[assemble_circuit(circuit, basis_gates_set) for circuit in circuits]
        )
    else:
        aer_circuits, idx_maps = zip(*[assemble_circuit(circuit) for circuit in circuits])
    return list(aer_circuits), list(idx_maps)
