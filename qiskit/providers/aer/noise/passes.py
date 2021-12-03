# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Noise addition passes.
"""
from typing import Optional, Union, Sequence, Callable, List, Iterable

import numpy as np

from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from .errors import QuantumError
from .errors.standard_errors import thermal_relaxation_error

InstructionLike = Union[Instruction, QuantumError]


class LocalNoisePass(TransformationPass):
    """Transpiler pass to insert noise into a circuit.

    The noise in this pass is defined by a noise function or callable with signature

    .. code:: python

            def fn(
                inst: Instruction,
                qubits: Optional[List[int]] = None
            ) -> InstructionLike:

    For every instance of one of the reference instructions in a circuit the
    supplied function is called on that instruction and the returned noise
    is added to the circuit. This noise can depend on properties of the
    instruction it is called on (for example parameters or duration) to
    allow inserting parameterized noise models.

    Several methods for adding the constructed errors to circuits are supported
    and can be set by using the ``method`` kwarg. The supported methods are

    * ``"append"``: add the return of the callable after the instruction.
    * ``"prepend"``: add the return of the callable before the instruction.
    * ``"replace"``: replace the instruction with the return of the callable.

    """

    def __init__(
            self,
            func: Callable[[Instruction, Sequence[int]], InstructionLike],
            ops: Optional[Union[Instruction, Iterable[Instruction]]] = None,
            method: str = 'append'
    ):
        """Initialize noise pass.

        Args:
            fn: noise function `fn(inst, qubits) -> InstructionLike`.
            ops: Optional, single or list of instructions to apply the
                                  noise function to. If None the noise function will be
                                  applied to all instructions in the circuit.
            method: method for inserting noise. Allow methods are
                           'append', 'prepend', 'replace'.
        Raises:
            TranspilerError: if an invalid option is specified.
        """
        if method not in {"append", "prepend", "replace"}:
            raise TranspilerError(
                f'Invalid method: {method}, it must be "append", "prepend" or "replace"'
            )
        if isinstance(ops, str):
            ops = [ops]
        super().__init__()
        self._fn = fn
        self._ops = set(ops) if ops else {}
        self._method = method

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the LocalNoisePass pass on `dag`.
        Args:
            dag: DAG to be changed.
        Returns:
            A changed DAG.
        Raises:
            TranspilerError: if generated operation is not valid.
        """
        qubit_indices = {qubit: idx for idx, qubit in enumerate(dag.qubits)}
        for node in dag.topological_op_nodes():
            if self._ops and node.op.name not in self._ops:
                continue

            qubits = [qubit_indices[q] for q in node.qargs]
            new_op = self._fn(node.op, qubits)
            if new_op is None:
                continue
            if not isinstance(new_op, Instruction):
                try:
                    new_op = new_op.to_instruction()
                except AttributeError as att_err:
                    raise TranspilerError(
                        "Function must return an object implementing 'to_instruction' method."
                    ) from att_err
            if new_op.num_qubits != len(node.qargs):
                raise TranspilerError(
                    f"Number of qubits of generated op {new_op.num_qubits} != "
                    f"{len(node.qargs)} that of a reference op {node.name}"
                )

            new_dag = DAGCircuit()
            new_dag.add_qubits(node.qargs)
            new_dag.add_clbits(node.cargs)
            if self._method == "append":
                new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
            new_dag.apply_operation_back(new_op, qargs=node.qargs)
            if self._method == "prepend":
                new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)

            dag.substitute_node_with_dag(node, new_dag)

        return dag


class RelaxationNoisePass(LocalNoisePass):
    """Add duration dependent thermal relaxation noise after instructions."""

    def __init__(
            self,
            t1s: List[float],
            t2s: List[float],
            dt: float,
            ops: Optional[Union[Instruction, Sequence[Instruction]]] = None,
            excited_state_populations: Optional[List[float]] = None,
    ):
        """Initialize RelaxationNoisePass.

        Args:
            t1s: List of T1 times in seconds for each qubit.
            t2s: List of T2 times in seconds for each qubit.
            dt: ...
            ops: Optional, the operations to add relaxation to. If None
                 relaxation will be added to all operations.
            excited_state_populations: Optional, list of excited state populations
                for each qubit at thermal equilibrium. If not supplied or obtained
                from the backend this will be set to 0 for each qubit.
        """
        self._t1s = np.asarray(t1s)
        self._t2s = np.asarray(t2s)
        if excited_state_populations is not None:
            self._p1s = np.asarray(excited_state_populations)
        else:
            self._p1s = np.zeros(len(t1s))
        self._dt = dt
        super().__init__(self._thermal_relaxation_error, ops=ops, method="append")

    def _thermal_relaxation_error(
            self,
            op: Instruction,
            qubits: Sequence[int]
    ):
        """Return thermal relaxation error on each gate qubit"""
        duration = op.duration
        if duration == 0:
            return None

        # convert time unit in seconds
        duration = duration * self._dt

        t1s = self._t1s[qubits]
        t2s = self._t2s[qubits]
        p1s = self._p1s[qubits]

        # pylint: disable=invalid-name
        if op.num_qubits == 1:
            t1, t2, p1 = t1s[0], t2s[0], p1s[0]
            if t1 == np.inf and t2 == np.inf:
                return None
            return thermal_relaxation_error(t1, t2, duration, p1)

        # General multi-qubit case
        noise = QuantumCircuit(op.num_qubits)
        for qubit, (t1, t2, p1) in enumerate(zip(t1s, t2s, p1s)):
            if t1 == np.inf and t2 == np.inf:
                # No relaxation on this qubit
                continue
            error = thermal_relaxation_error(t1, t2, duration, p1)
            noise.append(error, [qubit])

        return noise

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the RelaxationNoisePass pass on `dag`.
        Args:
            dag: DAG to be changed.
        Returns:
            A changed DAG.
        Raises:
            TranspilerError: if failed to insert noises to the dag.
        """
        if dag.duration is None:
            raise TranspilerError("This pass accepts only scheduled circuits")

        return super().run(dag)
