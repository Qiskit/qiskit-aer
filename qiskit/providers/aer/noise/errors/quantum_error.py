# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
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
import logging
import copy

import numpy as np

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators import Kraus, SuperOp, Choi, Operator
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT

from ..noiseerror import NoiseError
from .errorutils import kraus2instructions
from .errorutils import circuit2superop
from .errorutils import standard_instruction_channel
from .errorutils import standard_instruction_operator

logger = logging.getLogger(__name__)


class QuantumError:
    """
    Quantum error class for Qiskit Aer noise model

    WARNING: The init interface for this class is not finalized and may
             change in future releases. For maximum backwards compatibility
             use the QuantumError generating functions in the `noise.errors`
             module.
    """

    # pylint: disable=invalid-name
    _ATOL_DEFAULT = ATOL_DEFAULT
    _RTOL_DEFAULT = RTOL_DEFAULT
    _MAX_TOL = 1e-4

    def __init__(self,
                 noise_ops,
                 number_of_qubits=None,
                 standard_gates=True,
                 atol=ATOL_DEFAULT):
        """
        Create a quantum error for a noise model.

        Noise ops may either be specified as list of Kraus operators
        for a general CPTP map, or as a list of ``(circuit, p)`` pairs
        where ``circuit`` is a qobj circuit for the noise, and ``p`` is
        the probability of the error circuit. If the input is Kraus
        operators they will be converted to the circuit format, with
        checks applied for determining if any Kraus operators are
        unitary matrices.

        **Example**

        An example noise_ops for a bit-flip error with error probability
        ``p = 0.1`` is:

        .. code-block:: python

            noise_ops = [([{"name": "id", "qubits": 0}], 0.9),
                         ([{"name": "x", "qubits": 0}], 0.1)]

        The same error represented as a Kraus channel can be input as:

        .. code-block:: python

            noise_ops = [np.sqrt(0.9) * np.array([[1, 0], [0, 1]]),
                         np.sqrt(0.1) * np.array([[0, 1], [1, 0]])]

        Args:
            noise_ops (list): A list of noise ops. See additional information.
            number_of_qubits (int): specify the number of qubits for the
                                    error. If None this will be determined
                                    automatically (default None).
            standard_gates (bool): Check if input matrices are standard gates.
            atol (double): Threshold for testing if probabilities are
                           equal to 0 or 1 (Default: 1e-8).

        Raises:
            NoiseError: If input noise_ops are not a CPTP map.
        """

        # Shallow copy constructor
        if isinstance(noise_ops, QuantumError):
            self._number_of_qubits = noise_ops._number_of_qubits
            self._noise_circuits = noise_ops._noise_circuits
            self._noise_probabilities = noise_ops._noise_probabilities
            return

        # Initialize internal variables
        self._number_of_qubits = None
        self._noise_circuits = []
        self._noise_probabilities = []

        # Convert operator subclasses into Kraus list
        if issubclass(noise_ops.__class__, BaseOperator) or hasattr(
                noise_ops, 'to_quantumchannel') or hasattr(
                    noise_ops, 'to_operator'):
            noise_ops, other = Kraus(noise_ops)._data
            if other is not None:
                # A Kraus map with different left and right Kraus matrices
                # cannot be CPTP
                raise NoiseError("Input quantum channel is not CPTP.")

        # Convert iterable input into list
        noise_ops = list(noise_ops)
        # Check if Kraus
        if isinstance(noise_ops[0], np.ndarray):
            noise_ops = kraus2instructions(
                noise_ops, standard_gates, atol=atol)
        minimum_qubits = 0
        # Add non-zero probability error circuits to the error
        for circuit, prob in noise_ops:
            if prob > 0.0:
                self._noise_circuits.append(circuit)
                self._noise_probabilities.append(prob)
                # Determinine minimum qubit number for error from circuits
                for gate in circuit:
                    gate_qubits = max(gate["qubits"]) + 1
                    minimum_qubits = max([minimum_qubits, gate_qubits])

        # Set number of qubits
        if number_of_qubits is None:
            self._number_of_qubits = minimum_qubits
        else:
            self._number_of_qubits = number_of_qubits

        # Combine any kraus circuits
        noise_ops = self._combine_kraus(noise_ops, self.number_of_qubits)

        # Error checking
        if minimum_qubits > self._number_of_qubits:
            raise NoiseError("Input errors require {} qubits, "
                             "but number_of_qubits is {}".format(
                                 minimum_qubits, number_of_qubits))
        if len(self._noise_circuits) != len(self._noise_probabilities):
            raise NoiseError(
                "Number of error circuits does not match length of probabilities"
            )
        total_probs = np.sum(self._noise_probabilities)
        if abs(total_probs - 1) > atol:
            raise NoiseError(
                "Probabilities are not normalized: {} != 1".format(
                    total_probs))
        if [p for p in self._noise_probabilities if (isinstance(p, complex) or (p < 0))]:
            raise NoiseError("Probabilities are invalid.")
        # Rescale probabilities if their sum is ok to avoid
        # accumulation of rounding errors
        self._noise_probabilities = list(np.array(self._noise_probabilities) / total_probs)

    def __repr__(self):
        """Display QuantumError."""
        return "QuantumError({})".format(
            list(zip(self.circuits, self.probabilities)))

    def __str__(self):
        """Print error information."""
        output = "QuantumError on {} qubits. Noise circuits:".format(
            self._number_of_qubits)
        for j, pair in enumerate(zip(self.probabilities, self.circuits)):
            output += "\n  P({0}) = {1}, QasmQobjInstructions = [{2}".format(
                j, pair[0], pair[1])
        return output

    def __eq__(self, other):
        """Test if two QuantumErrors are equal as SuperOps"""
        if not isinstance(other, QuantumError):
            return False
        return self.to_quantumchannel() == other.to_quantumchannel()

    def copy(self):
        """Make a copy of current QuantumError."""
        # pylint: disable=no-value-for-parameter
        # The constructor of subclasses from raw data should be a copy
        return copy.deepcopy(self)

    @property
    def atol(self):
        """The default absolute tolerance parameter for float comparisons."""
        return QuantumError._ATOL_DEFAULT

    @property
    def rtol(self):
        """The relative tolerance parameter for float comparisons."""
        return QuantumError._RTOL_DEFAULT

    @classmethod
    def set_atol(cls, value):
        """Set the class default absolute tolerance parameter for float comparisons."""
        if value < 0:
            raise NoiseError(
                "Invalid atol ({}) must be non-negative.".format(value))
        if value > cls._MAX_TOL:
            raise NoiseError(
                "Invalid atol ({}) must be less than {}.".format(
                    value, cls._MAX_TOL))
        cls._ATOL_DEFAULT = value

    @classmethod
    def set_rtol(cls, value):
        """Set the class default relative tolerance parameter for float comparisons."""
        if value < 0:
            raise NoiseError(
                "Invalid rtol ({}) must be non-negative.".format(value))
        if value > cls._MAX_TOL:
            raise NoiseError(
                "Invalid rtol ({}) must be less than {}.".format(
                    value, cls._MAX_TOL))
        cls._RTOL_DEFAULT = value

    @property
    def size(self):
        """Return the number of error circuit."""
        return len(self._noise_circuits)

    @property
    def number_of_qubits(self):
        """Return the number of qubits for the error."""
        return self._number_of_qubits

    @property
    def circuits(self):
        """Return the list of error circuits."""
        return self._noise_circuits

    @property
    def probabilities(self):
        """Return the list of error probabilities."""
        return self._noise_probabilities

    def ideal(self):
        """Return True if current error object is an identity"""
        instructions, probability = self.error_term(0)
        if probability == 1 and instructions == [{
                "name": "id",
                "qubits": [0]
        }]:
            logger.debug("Error object is ideal")
            return True
        return False

    def to_quantumchannel(self):
        """Convert the QuantumError to a SuperOp quantum channel."""
        # Initialize as an empty superoperator of the correct size
        dim = 2**self.number_of_qubits
        channel = SuperOp(np.zeros([dim * dim, dim * dim]))
        for circuit, prob in zip(self._noise_circuits,
                                 self._noise_probabilities):
            component = prob * circuit2superop(circuit, self.number_of_qubits)
            channel = channel + component
        return channel

    def to_instruction(self):
        """Convert the QuantumError to a circuit Instruction."""
        return self.to_quantumchannel().to_instruction()

    def error_term(self, position):
        """
        Return a single term from the error.

        Args:
            position (int): the position of the error term.

        Returns:
            tuple: A pair `(p, circuit)` for error term at `position` < size
            where `p` is the probability of the error term, and `circuit`
            is the list of qobj instructions for the error term.

        Raises:
            NoiseError: If the position is greater than the size of
            the quantum error.
        """
        if position < self.size:
            return self.circuits[position], self.probabilities[position]
        else:
            raise NoiseError("Position {} is greater than the number".format(
                position) + "of error outcomes {}".format(self.size))

    def to_dict(self):
        """Return the current error as a dictionary."""
        error = {
            "type": "qerror",
            "operations": [],
            "instructions": list(self._noise_circuits),
            "probabilities": list(self._noise_probabilities)
        }
        return error

    def compose(self, other, front=False):
        """Return the composition error channel other * self.

        Note that for `front=True` this is equivalent to the
        :meth:`QuantumError.dot` method.

        Args:
            other (QuantumError): a quantum error channel.
            front (bool): If True return the reverse order composation
                          self * other instead [default: False].

        Returns:
            QuantumError: The composition error channel.

        Raises:
            NoiseError: if other cannot be converted into a QuantumError,
            or has incompatible dimensions.
        """
        if front:
            return self._matmul(other, left_multiply=False)
        return self._matmul(other, left_multiply=True)

    def dot(self, other):
        """Return the composition error channel self * other.

        Args:
            other (QuantumError): a quantum error channel.

        Returns:
            QuantumError: The composition error channel.

        Raises:
            NoiseError: if other cannot be converted into a QuantumError,
            or has incompatible dimensions.
        """
        return self._matmul(other, left_multiply=False)

    def power(self, n):
        """Return the compose of a error channel with itself n times.

        Args:
            n (int): the number of times to compose with self (n>0).

        Returns:
            QuantumError: the n-times composition error channel.

        Raises:
            NoiseError: if the power is not a positive integer.
        """
        if not isinstance(n, int) or n < 1:
            raise NoiseError("Can only power with positive integer powers.")
        ret = self.copy()
        for _ in range(1, n):
            ret = ret.compose(self)
        return ret

    def tensor(self, other):
        """Return the tensor product quantum error channel self ⊗ other.

        Args:
            other (QuantumError): a quantum error channel.

        Returns:
            QuantumError: the tensor product error channel self ⊗ other.

        Raises:
            NoiseError: if other cannot be converted to a QuantumError.
        """
        return self._tensor_product(other, reverse=False)

    def expand(self, other):
        """Return the tensor product quantum error channel self ⊗ other.

        Args:
            other (QuantumError): a quantum error channel.

        Returns:
            QuantumError: the tensor product error channel other ⊗ self.

        Raises:
            NoiseError: if other cannot be converted to a QuantumError.
        """
        return self._tensor_product(other, reverse=True)

    def _matmul(self, other, left_multiply=False):
        """Return the composition quantum error.

        Args:
            other (QuantumError): a quantum error.
            left_multiply (bool): If True return other * self
                                  If False return self * other [Default:False]
        Returns:
            QuantumError: The composition quantum error.

        Raises:
            NoiseError: if other cannot be converted into a QuantumError,
            or has incompatible dimensions.
        """
        # Convert other into a quantum error
        if not isinstance(other, QuantumError):
            other = QuantumError(other)
        # Error checking
        if self.number_of_qubits != other.number_of_qubits:
            raise NoiseError(
                "QuantumErrors are not defined on same number of qubits.")

        combined_noise_circuits = []
        combined_noise_probabilities = []

        # Combine subcircuits and probabilities
        if left_multiply:
            noise_ops0 = list(
                zip(self._noise_circuits, self._noise_probabilities))
            noise_ops1 = list(
                zip(other._noise_circuits, other._noise_probabilities))
        else:
            noise_ops0 = list(
                zip(other._noise_circuits, other._noise_probabilities))
            noise_ops1 = list(
                zip(self._noise_circuits, self._noise_probabilities))
        # Combine subcircuits and probabilities
        for circuit0, prob0 in noise_ops0:
            for circuit1, prob1 in noise_ops1:
                combined_noise_probabilities.append(prob0 * prob1)
                tmp_combined = circuit0 + circuit1

                # Fuse compatible ops to reduce noise operations:
                combined_circuit = [tmp_combined[0]]
                for instr in tmp_combined[1:]:
                    last_instr = combined_circuit[-1]
                    last_name = last_instr['name']
                    name = instr['name']
                    can_combine = (last_name in ['id', 'kraus', 'unitary'] or
                                   name in ['id', 'kraus', 'unitary'])
                    if (can_combine and self._check_instr(last_name) and
                            self._check_instr(name)):
                        combined_circuit[-1] = self._compose_instr(
                            last_instr, instr, self.number_of_qubits)
                    else:
                        # If they cannot be combined append the operation
                        combined_circuit.append(instr)
                # Check if circuit is empty and add identity
                if not combined_circuit:
                    combined_circuit.append({'name': 'id', 'qubits': [0]})
                # Add circuit
                combined_noise_circuits.append(combined_circuit)
        noise_ops = self._combine_kraus(
            zip(combined_noise_circuits, combined_noise_probabilities),
            self.number_of_qubits)
        return QuantumError(noise_ops)

    def _tensor_product(self, other, reverse=False):
        """Return the tensor product error channel.

        Args:
            other (QuantumError): a quantum channel subclass
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False
        Returns:
            QuantumError: the tensor product error channel.

        Raises:
            NoiseError: if other cannot be converted to a QuantumError.
        """
        # Convert other into a quantum error
        if not isinstance(other, QuantumError):
            other = QuantumError(other)

        combined_noise_circuits = []
        combined_noise_probabilities = []
        # Combine subcircuits and probabilities
        if reverse:
            shift_qubits = self.number_of_qubits
            noise_ops0 = list(
                zip(self._noise_circuits, self._noise_probabilities))
            noise_ops1 = list(
                zip(other._noise_circuits, other._noise_probabilities))
        else:
            shift_qubits = other.number_of_qubits
            noise_ops0 = list(
                zip(other._noise_circuits, other._noise_probabilities))
            noise_ops1 = list(
                zip(self._noise_circuits, self._noise_probabilities))
        for circuit1, prob1 in noise_ops1:
            for circuit0, prob0 in noise_ops0:
                combined_noise_probabilities.append(prob0 * prob1)
                # Shift qubits in circuit1
                circuit1_shift = []
                for instr in circuit1:
                    tmp = instr.copy()
                    tmp['qubits'] = [q + shift_qubits for q in tmp['qubits']]
                    circuit1_shift.append(tmp)
                tmp_combined = circuit0 + circuit1_shift
                # Fuse compatible ops to reduce noise operations:
                combined_circuit = [tmp_combined[0]]
                for instr in tmp_combined[1:]:
                    # Check if instructions can be combined
                    last_instr = combined_circuit[-1]
                    last_name = last_instr['name']
                    name = instr['name']
                    can_combine = (last_name in ['id', 'kraus', 'unitary'] or
                                   name in ['id', 'kraus', 'unitary'])
                    if (can_combine and self._check_instr(last_name) and
                            self._check_instr(name)):
                        combined_circuit[-1] = self._tensor_instr(
                            last_instr, instr)
                    else:
                        # If they cannot be combined append the operation
                        combined_circuit.append(instr)
                # Check if circuit is empty and add identity
                if not combined_circuit:
                    combined_circuit.append({'name': 'id', 'qubits': [0]})
                # Add circuit
                combined_noise_circuits.append(combined_circuit)
        # Now we combine any error circuits containing only Kraus operations
        noise_ops = self._combine_kraus(
            zip(combined_noise_circuits, combined_noise_probabilities),
            self.number_of_qubits + other.number_of_qubits)
        return QuantumError(noise_ops)

    @staticmethod
    def _combine_kraus(noise_ops, num_qubits):
        """Combine any noise circuits containing only Kraus instructions."""
        kraus_instr = []
        kraus_probs = []
        new_circuits = []
        new_probs = []
        # Partion circuits into Kraus and non-Kraus
        for circuit, prob in noise_ops:
            if len(circuit) == 1 and circuit[0]['name'] == 'kraus':
                kraus_instr.append(circuit[0])
                kraus_probs.append(prob)
            else:
                new_circuits.append(circuit)
                new_probs.append(prob)
        # Combine matching Kraus instructions via Choi rep
        if len(kraus_probs) == 1:
            new_circuits.append([kraus_instr[0]])
            new_probs.append(kraus_probs[0])
        elif len(kraus_probs) > 1:
            dim = 2 ** num_qubits
            iden = SuperOp(np.eye(dim ** 2))
            choi_sum = Choi(np.zeros((dim ** 2, dim ** 2)))
            for prob, instr in zip(kraus_probs, kraus_instr):
                choi_sum = choi_sum + prob * iden.compose(Kraus(instr['params']),
                                                          instr['qubits'])
            # Renormalize the Choi operator to find probability
            # of Kraus error
            chan_prob = abs(np.trace(choi_sum.data) / dim)
            chan_instr = {
                "name": "kraus",
                "qubits": list(range(num_qubits)),
                "params": Kraus(choi_sum / chan_prob).data
            }
            new_circuits.append([chan_instr])
            new_probs.append(chan_prob)
        return list(zip(new_circuits, new_probs))

    @staticmethod
    def _check_instr(name):
        """Check if instruction name can be converted to standard operator"""
        return name in [
            'kraus', 'unitary', 'reset', 'u1', 'u2', 'u3', 'id', 'x', 'y', 'z',
            'h', 's', 'sdg', 't', 'tdg', 'cx', 'cz', 'swap', 'ccx'
        ]

    @staticmethod
    def _instr2op(instr):
        """Try and convert an instruction into an operator"""
        # Try and convert to operator first
        operator = standard_instruction_operator(instr)
        if operator is not None:
            return operator
        # Otherwise return SuperOp or None
        return standard_instruction_channel(instr)

    @staticmethod
    def _tensor_instr(instr0, instr1):
        """Tensor of two operator qobj instructions."""
        # If one of the instructions is an identity we only need
        # to return the other instruction
        if instr0['name'] == 'id':
            return instr1
        if instr1['name'] == 'id':
            return instr0
        # Combine qubits
        qubits = instr0['qubits'] + instr1['qubits']
        # Convert to ops
        op0 = QuantumError._instr2op(instr0)
        op1 = QuantumError._instr2op(instr1)
        # Check if at least one of the instructions is a channel
        # and if so convert to Kraus representation.
        if isinstance(op0, SuperOp) or isinstance(op1, SuperOp):
            name = 'kraus'
            params = Kraus(SuperOp(op0).expand(op1)).data
        else:
            name = 'unitary'
            params = [op0.expand(op1).data]
        return {'name': name, 'qubits': qubits, 'params': params}

    @staticmethod
    def _compose_instr(instr0, instr1, num_qubits):
        """Helper function for compose a kraus with another instruction."""
        # If one of the instructions is an identity we only need
        # to return the other instruction
        if instr0['name'] == 'id':
            return instr1
        if instr1['name'] == 'id':
            return instr0
        # Convert to ops
        op0 = QuantumError._instr2op(instr0)
        op1 = QuantumError._instr2op(instr1)
        # Check if at least one of the instructions is a channel
        # and if so convert both to SuperOp representation
        if isinstance(op0,
                      (SuperOp, Kraus)) or isinstance(op1, (SuperOp, Kraus)):
            name = 'kraus'
            op0 = SuperOp(op0)
            op1 = SuperOp(op1)
        else:
            name = 'unitary'
        # Check qubits for compositions
        qubits0 = instr0['qubits']
        qubits1 = instr1['qubits']
        if qubits0 == qubits1:
            composed = op0.compose(op1)
            qubits = qubits0
        else:
            # If qubits don't match we compose with total number of qubits
            # for the error
            if name == 'kraus':
                composed = SuperOp(np.eye(4 ** num_qubits))
            else:
                composed = Operator(np.eye(2 ** num_qubits))
            composed = composed.compose(op0, qargs=qubits0).compose(op1, qargs=qubits1)
            qubits = list(range(num_qubits))
        # Get instruction params
        if name == 'kraus':
            params = Kraus(composed).data
        else:
            params = [composed.data]
        return {'name': name, 'qubits': qubits, 'params': params}

    # Overloads
    def __matmul__(self, other):
        return self.compose(other)

    def __mul__(self, other):
        return self.dot(other)

    def __pow__(self, n):
        return self.power(n)

    def __xor__(self, other):
        return self.tensor(other)

    def __rmul__(self, other):
        raise NotImplementedError(
            "'QuantumError' does not support scalar multiplication.")

    def __truediv__(self, other):
        raise NotImplementedError("'QuantumError' does not support division.")

    def __add__(self, other):
        raise NotImplementedError("'QuantumError' does not support addition.")

    def __sub__(self, other):
        raise NotImplementedError(
            "'QuantumError' does not support subtraction.")

    def __neg__(self):
        raise NotImplementedError("'QuantumError' does not support negation.")
