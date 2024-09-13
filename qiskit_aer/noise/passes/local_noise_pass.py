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
Local noise addition pass.
"""
from typing import Optional, Union, Sequence, Callable, Iterable

from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from ..errors import QuantumError, ReadoutError

InstructionLike = Union[Instruction, QuantumError, QuantumCircuit]


class LocalNoisePass(TransformationPass):
    """Transpiler pass to insert noise into a circuit.

    The noise in this pass is defined by a noise function or callable with signature

    .. code:: python

            def func(
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
        If the return is None, the instruction will be removed.

    """

    def __init__(
        self,
        func: Callable[[Instruction, Sequence[int]], Optional[InstructionLike]],
        op_types: Optional[Union[type, Iterable[type]]] = None,
        method: str = "append",
    ):
        """Initialize noise pass.

        Args:
            func: noise function `func(inst, qubits) -> InstructionLike`.
            op_types: Optional, single or list of instruction types to apply the
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
        if isinstance(op_types, type):
            op_types = (op_types,)
        super().__init__()
        self._func = func
        self._ops = tuple(op_types) if op_types else tuple()
        self._method = method
        if not all(isinstance(op, type) for op in self._ops):
            raise TranspilerError(
                f"Invalid ops: '{op_types}', expecting single or list of operation types (or None)"
            )

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
            if self._ops and not isinstance(node.op, self._ops):
                continue

            qubits = [qubit_indices[q] for q in node.qargs]
            new_op = self._func(node.op, qubits)

            if new_op is None:
                # Edge case where we are replacing a node with nothing (removing node)
                if self._method == "replace":
                    dag.remove_op_node(node)
                continue

            if isinstance(new_op, ReadoutError):
                raise TranspilerError("Insertions of ReadoutError is not yet supported.")

            # Initialize new node dag
            new_dag = DAGCircuit()
            new_dag.add_qubits(node.qargs)
            new_dag.add_clbits(node.cargs)

            # If appending re-apply original op node first
            if self._method == "append":
                new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)

            # If the new op is not a QuantumCircuit or Instruction, attempt
            # to conver to an Instruction
            if not isinstance(new_op, (QuantumCircuit, Instruction)):
                try:
                    new_op = new_op.to_instruction()
                except AttributeError as att_err:
                    raise TranspilerError(
                        "Function must return an object implementing 'to_instruction' method."
                    ) from att_err

            if new_op.num_clbits > 0:
                raise TranspilerError("Noise must be an instruction without clbits.")

            # Validate the instruction matches the number of qubits and clbits of the node
            if new_op.num_qubits != len(node.qargs):
                raise TranspilerError(
                    f"Number of qubits of generated op {new_op.num_qubits} != "
                    f"{len(node.qargs)} that of a reference op {node.name}"
                )

            # Add the noise op returned by the function
            if isinstance(new_op, QuantumCircuit):
                # If the new op is a quantum circuit, compose its DAG with the new dag
                # so that it is unrolled rather than added as an opaque instruction
                new_dag.compose(
                    circuit_to_dag(new_op), qubits=list(node.qargs)
                )  # never touch clbits
            else:
                # Otherwise append the instruction returned by the function
                new_dag.apply_operation_back(new_op, qargs=node.qargs)  # never touch cargs

            # If prepending reapply original op node last
            if self._method == "prepend":
                new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)

            dag.substitute_node_with_dag(node, new_dag)

        return dag
