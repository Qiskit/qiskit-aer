# This code is part of Qiskit.
#
# (C) Copyright IBM 2018-2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Quantum error class for Aer noise model
"""
import numbers
from typing import Iterable
import numpy as np

from qiskit.circuit import QuantumCircuit, Instruction, Reset
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library.generalized_gates import PauliGate, UnitaryGate
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Kraus, SuperOp, Clifford, Pauli
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.mixins import TolerancesMixin
from qiskit.quantum_info.operators.predicates import is_identity_matrix

from .base_quantum_error import BaseQuantumError
from ..noiseerror import NoiseError


class QuantumError(BaseQuantumError, TolerancesMixin):
    """
    Quantum error class for Aer noise model

    .. warning::
             The init interface for this class is not finalized and may
             change in future releases. For maximum backwards compatibility
             use the QuantumError generating functions in the `noise.errors`
             module.
    """

    def __init__(self, noise_ops):
        """
        Create a quantum error for a noise model.

        Noise ops may either be specified as a
        :obj:`~qiskit.quantum_info.operators.channel.quantum_channel.QuantumChannel`
        for a general CPTP map, or as a list of ``(circuit, p)`` pairs
        where ``circuit`` is a circuit-like object for the noise, and
        ``p`` is the probability of the noise event. Any type of input
        will be converted to the probabilistic mixture of circuit format.

        **Example**

        An example noise_ops for a bit-flip error with error probability
        ``p = 0.1`` is:

        .. code-block:: python

            noise_ops = [(IGate(), 0.9),
                         (XGate(), 0.1)]

        or specifying explicit qubit arguments,

        .. code-block:: python

            noise_ops = [((IGate(), [0]), 0.9),
                         ((XGate(), [0]), 0.1)]

        The same error represented as a Kraus channel can be input as:

        .. code-block:: python

            noise_ops = Kraus([np.sqrt(0.9) * np.array([[1, 0], [0, 1]]),
                               np.sqrt(0.1) * np.array([[0, 1], [1, 0]])])

        Args:
            noise_ops (QuantumChannel or Iterable): Either a quantum channel or a list of
                ``(circuit, p)`` pairs, which represents a quantum error, where
                ``circuit`` is a circuit-like object for the noise, and
                ``p`` is the probability of the noise event. Circuit-like types include
                ``QuantumCircuit``, ``(Instruction, qargs)`` and a list of ``(Instruction, qargs)``.
                Note that ``qargs`` should be a list of integers and can be omitted
                (default qubits are used in that case). See also examples above.
        Raises:
            NoiseError: If input noise_ops is invalid, e.g. it's not a CPTP map.
        """
        # Shallow copy constructor
        if isinstance(noise_ops, QuantumError):
            self._circs = noise_ops.circuits
            self._probs = noise_ops.probabilities
            super().__init__(num_qubits=noise_ops.num_qubits)
            return

        # Single circuit case
        if not isinstance(noise_ops, Iterable) or (
            isinstance(noise_ops, tuple) and isinstance(noise_ops[0], Instruction)
        ):
            noise_ops = [(noise_ops, 1.0)]

        # Convert zipped object to list (to enable multiple iteration over it)
        if not isinstance(noise_ops, list):
            noise_ops = list(noise_ops)

        # Input checks
        for pair in noise_ops:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise NoiseError(f"Invalid type of input is found around '{pair}'")
            _, p = pair  # pylint: disable=invalid-name
            if not isinstance(p, numbers.Real):
                raise NoiseError(f"Invalid type of probability: {p}")
            if p < -1 * QuantumError.atol:
                raise NoiseError(f"Negative probability is invalid: {p}")

        # Remove zero probability circuits
        noise_ops = [(op, prob) for op, prob in noise_ops if prob > 0]

        if len(noise_ops) == 0:
            raise NoiseError(
                "noise_ops must contain at least one operator with non-zero probability"
            )

        ops, probs = zip(*noise_ops)  # unzip

        # Initialize internal variables with error checking
        total_probs = sum(probs)
        if not np.isclose(total_probs - 1, 0, atol=QuantumError.atol):
            raise NoiseError(f"Probabilities are not normalized: {total_probs} != 1")
        # Rescale probabilities if their sum is ok to avoid accumulation of rounding errors
        self._probs = list(np.array(probs) / total_probs)

        # Convert instructions to circuits
        circs = [self._to_circuit(op) for op in ops]

        num_qubits = max(qc.num_qubits for qc in circs)
        self._circs = [self._enlarge_qreg(qc, num_qubits) for qc in circs]

        # Check validity of circuits
        for circ in self._circs:
            if circ.clbits:
                raise NoiseError("Circuit with classical register cannot be a channel")
            if circ.num_qubits != num_qubits:
                raise NoiseError("Number of qubits used in noise ops must be the same")

        super().__init__(num_qubits=num_qubits)

    # pylint: disable=too-many-return-statements
    @classmethod
    def _to_circuit(cls, op):
        if isinstance(op, QuantumCircuit):
            return op
        if isinstance(op, tuple):
            inst, qubits = op
            circ = QuantumCircuit(max(qubits) + 1)
            circ.append(inst, qargs=qubits)
            return circ
        if isinstance(op, Instruction):
            if op.num_clbits > 0:
                raise NoiseError(
                    f"Unable to convert instruction with clbits: {op.__class__.__name__}"
                )
            circ = QuantumCircuit(op.num_qubits)
            circ.append(op, qargs=list(range(op.num_qubits)))
            return circ
        if isinstance(op, QuantumChannel):
            if not op.is_cptp(atol=cls.atol):
                raise NoiseError("Input quantum channel is not CPTP.")
            try:
                return cls._to_circuit(Kraus(op).to_instruction())
            except QiskitError as err:
                raise NoiseError(
                    f"Fail to convert {op.__class__.__name__} to Instruction."
                ) from err
        if isinstance(op, BaseOperator):
            if hasattr(op, "to_instruction"):
                try:
                    return cls._to_circuit(op.to_instruction())
                except QiskitError as err:
                    raise NoiseError(
                        f"Fail to convert {op.__class__.__name__} to Instruction."
                    ) from err
            else:
                raise NoiseError(
                    f"Unacceptable Operator, not implementing to_instruction: "
                    f"{op.__class__.__name__}"
                )
        if isinstance(op, list):
            if all(isinstance(aop, tuple) for aop in op):
                num_qubits = max(max(qubits) for _, qubits in op) + 1
                circ = QuantumCircuit(num_qubits)
                for inst, qubits in op:
                    try:
                        circ.append(inst, qargs=qubits)
                    except CircuitError as err:
                        raise NoiseError(
                            f"Invalid operation type: {inst.__class__.__name__},"
                            f" not appendable to circuit."
                        ) from err
                return circ
            else:
                raise NoiseError(f"Invalid type of op list: {op}")

        raise NoiseError(f"Invalid noise op type {op.__class__.__name__}: {op}")

    def __repr__(self):
        """Display QuantumError."""
        return (
            f"<{self._repr_name()}, num_qubits={self.num_qubits},"
            f" size={self.size}, probabilities={self.probabilities}>"
        )

    def __str__(self):
        """Print error information."""
        output = f"QuantumError on {self.num_qubits} qubits. Noise circuits:"
        for j, pair in enumerate(zip(self.probabilities, self.circuits)):
            output += f"\n  P({j}) = {pair[0]}, Circuit = \n{pair[1]}"
        return output

    def __eq__(self, other):
        """Test if two QuantumErrors are equal as SuperOps"""
        if not isinstance(other, BaseQuantumError):
            return False
        return self.to_quantumchannel() == other.to_quantumchannel()

    @property
    def size(self):
        """Return the number of error circuit."""
        return len(self.circuits)

    @property
    def circuits(self):
        """Return the list of error circuits."""
        return self._circs

    @property
    def probabilities(self):
        """Return the list of error probabilities."""
        return self._probs

    def ideal(self):
        """Return True if this error object is composed only of identity operations.
        Note that the identity check is best effort and up to global phase."""
        for circ in self.circuits:
            try:
                # Circuit-level identity check for clifford Circuits
                clifford = Clifford(circ)
                if clifford != Clifford(np.eye(2 * circ.num_qubits, dtype=bool)):
                    return False
            except QiskitError:
                pass

            # Component-wise check for non-Clifford circuits
            for instruction in circ:
                op = instruction.operation
                if isinstance(op, IGate):
                    continue
                if isinstance(op, PauliGate):
                    if op.params[0].replace("I", ""):
                        return False
                else:
                    # Convert to Kraus and check if identity
                    kmats = Kraus(op).data
                    if len(kmats) > 1:
                        return False
                    if not is_identity_matrix(
                        kmats[0], ignore_phase=True, atol=self.atol, rtol=self.rtol
                    ):
                        return False
        return True

    def to_quantumchannel(self):
        """Convert the QuantumError to a SuperOp quantum channel.
        Required to enable SuperOp(QuantumError)."""
        # Initialize as an empty superoperator of the correct size
        dim = 2**self.num_qubits
        ret = SuperOp(np.zeros([dim * dim, dim * dim]))
        for circ, prob in zip(self.circuits, self.probabilities):
            component = prob * SuperOp(circ)
            ret = ret + component
        return ret

    def error_term(self, position):
        """
        Return a single term from the error.

        Args:
            position (int): the position of the error term.

        Returns:
            tuple: A pair `(circuit, p)` for error term at `position` < size
            where `p` is the probability of the error term, and `circuit`
            is the list of qobj instructions for the error term.

        Raises:
            NoiseError: If the position is greater than the size of
            the quantum error.
        """
        if position < self.size:
            return self.circuits[position], self.probabilities[position]
        else:
            raise NoiseError(
                f"Position {position} is greater than the number of error outcomes {self.size}"
            )

    def to_dict(self):
        """Return the current error as a dictionary."""
        # Assemble noise circuits
        instructions = []
        for circ in self._circs:
            circ_inst = []
            for inst in circ.data:
                inst_dict = {}
                inst_dict["name"] = inst.operation.name
                inst_dict["qubits"] = [circ.find_bit(q).index for q in inst.qubits]
                if inst.operation.params:
                    inst_dict["params"] = inst.operation.params
                if inst.operation.label:
                    inst_dict["label"] = inst.operation.label
                condition = getattr(inst.operation, "condition", None)
                if condition:
                    inst_dict["condition"] = condition
                circ_inst.append(inst_dict)
            instructions.append(circ_inst)
        # Construct error dict
        error = {
            "type": "qerror",
            "id": self.id,
            "operations": [],
            "instructions": instructions,
            "probabilities": list(self.probabilities),
        }
        return error

    @staticmethod
    def from_dict(error):
        """Implement current error from a dictionary."""
        # check if dictionary
        if not isinstance(error, dict):
            raise NoiseError("error is not a dictionary")
        # check expected keys "type, id, operations, instructions, probabilities"
        if (
            ("type" not in error)
            or ("id" not in error)
            or ("operations" not in error)
            or ("instructions" not in error)
            or ("probabilities" not in error)
        ):
            raise NoiseError("error dictionary not containing expected keys")
        error_instructions = error["instructions"]
        error_probabilities = error["probabilities"]

        if len(error_instructions) != len(error_probabilities):
            raise NoiseError("probabilities not matching with instructions")
        # parse instructions and turn to noise_ops
        noise_ops = []
        for idx, inst in enumerate(error_instructions):
            noise_elem = []
            for elem in inst:
                inst_name = elem["name"]
                inst_qubits = elem["qubits"]
                if inst_name == "x":
                    noise_elem.append((XGate(), inst_qubits))
                elif inst_name == "id":
                    noise_elem.append((IGate(), inst_qubits))
                elif inst_name == "y":
                    noise_elem.append((YGate(), inst_qubits))
                elif inst_name == "z":
                    noise_elem.append((ZGate(), inst_qubits))
                elif inst_name == "reset":
                    noise_elem.append((Reset(), inst_qubits))
                elif inst_name == "measure":
                    raise NoiseError("instruction 'measure' not supported")
                elif inst_name == "pauli":
                    if "params" not in elem:
                        raise NoiseError("pauli does not have a parameter value")
                    noise_elem.append((Pauli(elem["params"][0]), inst_qubits))
                elif inst_name == "kraus":
                    if "params" not in elem:
                        raise NoiseError("kraus does not have a parameter value")
                    noise_elem.append((Kraus(elem["params"]), inst_qubits))
                elif inst_name == "unitary":
                    if "params" not in elem:
                        raise NoiseError("unitary does not have a parameter value")
                    noise_elem.append((UnitaryGate(elem["params"][0]), inst_qubits))
                else:
                    raise NoiseError("error gate for instruction not recognized")

            noise_ops.append((noise_elem, error_probabilities[idx]))

        error_obj = QuantumError(noise_ops)

        return error_obj

    def compose(self, other, qargs=None, front=False):
        if not isinstance(other, QuantumError):
            other = QuantumError(other)
        if qargs is not None:
            if self.num_qubits < other.num_qubits:
                raise QiskitError(
                    "Number of qubits of this error must be less than"
                    " that of the error to be composed if using 'qargs' argument."
                )
            if len(qargs) != other.num_qubits:
                raise QiskitError(
                    "Number of items in 'qargs' argument must be the same as"
                    " number of qubits of the error to be composed."
                )
            if front:
                raise QiskitError(
                    "QuantumError.compose does not support 'qargs' when 'front=True'."
                )

        circs = [
            self._compose_circ(lqc, rqc, qubits=qargs, front=front)
            for lqc in self.circuits
            for rqc in other.circuits
        ]
        probs = [lpr * rpr for lpr in self.probabilities for rpr in other.probabilities]
        return QuantumError(zip(circs, probs))

    @staticmethod
    def _enlarge_qreg(qc: QuantumCircuit, num_qubits: int):
        if qc.num_qubits < num_qubits:
            enlarged = QuantumCircuit(num_qubits)
            return enlarged.compose(qc)
        return qc

    @staticmethod
    def _compose_circ(lqc: QuantumCircuit, rqc: QuantumCircuit, qubits, front):
        if qubits is None:
            if front:
                lqc, rqc = rqc, lqc
            if lqc.num_qubits < rqc.num_qubits:
                lqc = QuantumError._enlarge_qreg(lqc, rqc.num_qubits)
            return lqc.compose(rqc)

        return lqc.compose(rqc, qubits=qubits, front=front)

    def tensor(self, other):
        if not isinstance(other, QuantumError):
            other = QuantumError(other)

        circs = [lqc.tensor(rqc) for lqc in self.circuits for rqc in other.circuits]
        probs = [lpr * rpr for lpr in self.probabilities for rpr in other.probabilities]
        return QuantumError(zip(circs, probs))

    def expand(self, other):
        return other.tensor(self)
