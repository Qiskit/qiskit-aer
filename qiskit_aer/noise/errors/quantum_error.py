# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Quantum error class for Qiskit Aer noise model
"""
import copy
import numbers
import uuid
import warnings
from typing import Iterable

import numpy as np

from qiskit.circuit import QuantumCircuit, Instruction, QuantumRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.circuit.library.standard_gates import IGate
from qiskit.exceptions import QiskitError
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel import Kraus, SuperOp
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.mixins import TolerancesMixin
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from qiskit.quantum_info.operators.symplectic import Clifford
from .errorutils import _standard_gates_instructions
from .errorutils import kraus2instructions
from .errorutils import standard_gate_unitary
from ..noiseerror import NoiseError


class QuantumError(BaseOperator, TolerancesMixin):
    """
    Quantum error class for Qiskit Aer noise model

    .. warning::
             The init interface for this class is not finalized and may
             change in future releases. For maximum backwards compatibility
             use the QuantumError generating functions in the `noise.errors`
             module.
    """

    def __init__(self,
                 noise_ops,
                 number_of_qubits=None,
                 standard_gates=None,
                 atol=None):
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
            number_of_qubits (int): [DEPRECATED] specify the number of qubits for the
                                    error. If None this will be determined
                                    automatically (default None).
            standard_gates (bool): [DEPRECATED] Check if input matrices are standard gates.
            atol (double): [DEPRECATED] Threshold for testing if probabilities are
                           equal to 0 or 1 (Default: ``QuantumError.atol``).
        Raises:
            NoiseError: If input noise_ops is invalid, e.g. it's not a CPTP map.
        """
        # Unique ID for QuantumError
        self._id = uuid.uuid4().hex

        # Shallow copy constructor
        if isinstance(noise_ops, QuantumError):
            self._circs = noise_ops.circuits
            self._probs = noise_ops.probabilities
            super().__init__(num_qubits=noise_ops.num_qubits)
            return

        if atol is not None:
            warnings.warn(
                '"atol" option in the constructor of QuantumError has been deprecated'
                ' as of qiskit-aer 0.10.0 and will be removed no earlier than 3 months'
                ' from that release date. Use QuantumError.atol = value instead.',
                DeprecationWarning, stacklevel=2)
        else:
            atol = QuantumError.atol

        # Convert list of arrarys to kraus instruction (for old API support) TODO: to be removed
        if isinstance(noise_ops, (list, tuple)) and \
                len(noise_ops) > 0 and isinstance(noise_ops[0], np.ndarray):
            warnings.warn(
                'Constructing QuantumError with list of arrays representing a Kraus channel'
                ' has been deprecated as of qiskit-aer 0.10.0 and will be removed no earlier than'
                ' 3 months from that release date. Use QuantumError(Kraus(mats)) instead.',
                DeprecationWarning, stacklevel=2)
            if standard_gates:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",
                                            category=DeprecationWarning,
                                            module="qiskit_aer.noise.errors.errorutils")
                    noise_ops = kraus2instructions(
                        noise_ops, standard_gates, atol=atol)
            else:
                try:
                    noise_ops = Kraus(noise_ops)
                    noise_ops = [((noise_ops.to_instruction(),
                                   list(range(noise_ops.num_qubits))), 1.0)]
                except QiskitError as err:
                    raise NoiseError("Fail to convert Kraus to Instruction") from err

        # Single circuit case
        if not isinstance(noise_ops, Iterable) or \
                (isinstance(noise_ops, tuple) and isinstance(noise_ops[0], Instruction)):
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
            if p < -1 * atol:
                raise NoiseError(f"Negative probability is invalid: {p}")

        # Remove zero probability circuits
        noise_ops = [(op, prob) for op, prob in noise_ops if prob > 0]

        if len(noise_ops) == 0:
            raise NoiseError(
                "noise_ops must contain at least one operator with non-zero probability"
            )

        ops, probs = zip(*noise_ops)  # unzip

        if standard_gates is not None:
            warnings.warn(
                '"standard_gates" option in the constructor of QuantumError has been deprecated'
                ' as of qiskit-aer 0.10.0 in favor of externalizing such an unrolling functionality'
                ' and will be removed no earlier than 3 months from that release date.',
                DeprecationWarning, stacklevel=2)
            if standard_gates:
                if isinstance(ops[0], list):
                    ops = [_standard_gates_instructions(op) for op in ops]

        # Initialize internal variables with error checking
        total_probs = sum(probs)
        if not np.isclose(total_probs - 1, 0, atol=atol):
            raise NoiseError(f"Probabilities are not normalized: {total_probs} != 1")
        # Rescale probabilities if their sum is ok to avoid accumulation of rounding errors
        self._probs = list(np.array(probs) / total_probs)

        # Convert instructions to circuits
        circs = [self._to_circuit(op) for op in ops]

        num_qubits = max(qc.num_qubits for qc in circs)
        if number_of_qubits is not None:
            num_qubits = number_of_qubits
            warnings.warn(
                '"number_of_qubits" in the constructor of QuantumError has been deprecated'
                ' as of qiskit-aer 0.10.0 in favor of determining it automatically'
                ' and will be removed no earlier than 3 months from that release date.'
                ' Specify number of qubits in the quantum circuit passed to the init if necessary.',
                DeprecationWarning, stacklevel=2)
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
            if hasattr(op, 'to_instruction'):
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
                num_qubits = max([max(qubits) for _, qubits in op]) + 1
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
            # Support for old-style json-like input TODO: to be removed
            elif all(isinstance(aop, dict) for aop in op):
                warnings.warn(
                    'Constructing QuantumError with list of dict representing a mixed channel'
                    ' has been deprecated as of qiskit-aer 0.10.0 and will be removed'
                    ' no earlier than 3 months from that release date.',
                    DeprecationWarning, stacklevel=3)
                # Convert json-like to non-kraus Instruction
                num_qubits = max([max(dic['qubits']) for dic in op]) + 1
                circ = QuantumCircuit(num_qubits)
                for dic in op:
                    if dic['name'] == 'reset':
                        # pylint: disable=import-outside-toplevel
                        from qiskit.circuit import Reset
                        circ.append(Reset(), qargs=dic['qubits'])
                    elif dic['name'] == 'kraus':
                        circ.append(Instruction(name='kraus',
                                                num_qubits=len(dic['qubits']),
                                                num_clbits=0,
                                                params=dic['params']),
                                    qargs=dic['qubits'])
                    elif dic['name'] == 'unitary':
                        circ.append(UnitaryGate(data=dic['params'][0]),
                                    qargs=dic['qubits'])
                    elif dic['name'] == 'pauli':
                        circ.append(PauliGate(dic['params'][0]),
                                    qargs=dic['qubits'])
                    else:
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                category=DeprecationWarning,
                                module="qiskit_aer.noise.errors.errorutils"
                            )
                            circ.append(UnitaryGate(label=dic['name'],
                                                    data=standard_gate_unitary(dic['name'])),
                                        qargs=dic['qubits'])
                return circ
            else:
                raise NoiseError(f"Invalid type of op list: {op}")

        raise NoiseError(f"Invalid noise op type {op.__class__.__name__}: {op}")

    def __repr__(self):
        """Display QuantumError."""
        return f"QuantumError({list(zip(self.circuits, self.probabilities))})"

    def __str__(self):
        """Print error information."""
        output = f"QuantumError on {self.num_qubits} qubits. Noise circuits:"
        for j, pair in enumerate(zip(self.probabilities, self.circuits)):
            output += f"\n  P({j}) = {pair[0]}, Circuit = \n{pair[1]}"
        return output

    def __eq__(self, other):
        """Test if two QuantumErrors are equal as SuperOps"""
        if not isinstance(other, QuantumError):
            return False
        return self.to_quantumchannel() == other.to_quantumchannel()

    def __hash__(self):
        return hash(self._id)

    @property
    def id(self):   # pylint: disable=invalid-name
        """Return unique ID string for error"""
        return self._id

    def copy(self):
        """Make a copy of current QuantumError."""
        # pylint: disable=no-value-for-parameter
        # The constructor of subclasses from raw data should be a copy
        return copy.deepcopy(self)

    @classmethod
    def set_atol(cls, value):
        """Set the class default absolute tolerance parameter for float comparisons."""
        warnings.warn(
            'QuantumError.set_atol(value) has been deprecated as of qiskit-aer 0.10.0'
            ' and will be removed no earlier than 3 months from that release date.'
            ' Use QuantumError.atol = value instead.',
            DeprecationWarning, stacklevel=2)
        QuantumError.atol = value

    @classmethod
    def set_rtol(cls, value):
        """Set the class default relative tolerance parameter for float comparisons."""
        warnings.warn(
            'QuantumError.set_rtol(value) has been deprecated as of qiskit-aer 0.10.0'
            ' and will be removed no earlier than 3 months from that release date.'
            ' Use QuantumError.rtol = value instead.',
            DeprecationWarning, stacklevel=2)
        QuantumError.rtol = value

    @property
    def size(self):
        """Return the number of error circuit."""
        return len(self.circuits)

    @property
    def number_of_qubits(self):
        """Return the number of qubits for the error."""
        warnings.warn(
            "The `number_of_qubits` property has been deprecated as of"
            " qiskit-aer 0.10.0. Use the `num_qubits` property instead.",
            DeprecationWarning)
        return self.num_qubits

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
            for op, _, _ in circ:
                if isinstance(op, IGate):
                    continue
                if isinstance(op, PauliGate):
                    if op.params[0].replace('I', ''):
                        return False
                else:
                    # Convert to Kraus and check if identity
                    kmats = Kraus(op).data
                    if len(kmats) > 1:
                        return False
                    if not is_identity_matrix(kmats[0], ignore_phase=True,
                                              atol=self.atol, rtol=self.rtol):
                        return False
        return True

    def to_quantumchannel(self):
        """Convert the QuantumError to a SuperOp quantum channel.
        Required to enable SuperOp(QuantumError)."""
        # Initialize as an empty superoperator of the correct size
        dim = 2 ** self.num_qubits
        ret = SuperOp(np.zeros([dim * dim, dim * dim]))
        for circ, prob in zip(self.circuits, self.probabilities):
            component = prob * SuperOp(circ)
            ret = ret + component
        return ret

    def to_instruction(self):
        """Convert the QuantumError to a circuit Instruction."""
        return QuantumChannelInstruction(self)

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
            for inst, qargs, _ in circ.data:
                qobj_inst = inst.assemble()
                qobj_inst.qubits = [circ.find_bit(q).index for q in qargs]
                circ_inst.append(qobj_inst.to_dict())
            instructions.append(circ_inst)
        # Construct error dict
        error = {
            "type": "qerror",
            "id": self.id,
            "operations": [],
            "instructions": instructions,
            "probabilities": list(self.probabilities)
        }
        return error

    def compose(self, other, qargs=None, front=False):
        if not isinstance(other, QuantumError):
            other = QuantumError(other)
        if qargs is not None:
            if self.num_qubits < other.num_qubits:
                raise QiskitError("Number of qubits of this error must be less than"
                                  " that of the error to be composed if using 'qargs' argument.")
            if len(qargs) != other.num_qubits:
                raise QiskitError("Number of items in 'qargs' argument must be the same as"
                                  " number of qubits of the error to be composed.")
            if front:
                raise QiskitError(
                    "QuantumError.compose does not support 'qargs' when 'front=True'."
                )

        circs = [self._compose_circ(lqc, rqc, qubits=qargs, front=front)
                 for lqc in self.circuits
                 for rqc in other.circuits]
        probs = [lpr * rpr
                 for lpr in self.probabilities
                 for rpr in other.probabilities]
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

        circs = [lqc.tensor(rqc)
                 for lqc in self.circuits
                 for rqc in other.circuits]
        probs = [lpr * rpr
                 for lpr in self.probabilities
                 for rpr in other.probabilities]
        return QuantumError(zip(circs, probs))

    def expand(self, other):
        return other.tensor(self)

    # Overloads
    def __rmul__(self, other):
        raise NotImplementedError("'QuantumError' does not support scalar multiplication.")

    def __truediv__(self, other):
        raise NotImplementedError("'QuantumError' does not support division.")

    def __add__(self, other):
        raise NotImplementedError("'QuantumError' does not support addition.")

    def __sub__(self, other):
        raise NotImplementedError("'QuantumError' does not support subtraction.")

    def __neg__(self):
        raise NotImplementedError("'QuantumError' does not support negation.")


class QuantumChannelInstruction(Instruction):
    """Container instruction for adding QuantumError to circuit"""

    def __init__(self, quantum_error):
        """Initialize a quantum error circuit instruction.

        Args:
            quantum_error (QuantumError): the error to add as an instruction.
        """
        super().__init__("quantum_channel", quantum_error.num_qubits, 0, [])
        self._quantum_error = quantum_error

    def _define(self):
        """Allow unrolling to a Kraus instruction"""
        q = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(q, name=self.name)
        qc._append(Kraus(self._quantum_error).to_instruction(), q, [])
        self.definition = qc
