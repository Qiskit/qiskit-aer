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
Readout error class for Qiskit Aer noise model.
"""

import copy

import numpy as np
from numpy.linalg import norm

from qiskit.circuit import Instruction
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT

from ..noiseerror import NoiseError
from .errorutils import qubits_from_mat


class ReadoutError:
    """
    Readout error class for Qiskit Aer noise model.
    """
    # pylint: disable=invalid-name
    _ATOL_DEFAULT = ATOL_DEFAULT
    _RTOL_DEFAULT = RTOL_DEFAULT
    _MAX_TOL = 1e-4

    def __init__(self, probabilities, atol=ATOL_DEFAULT):
        """
        Create a readout error for a noise model.

        For an N-qubit readout error probabilities are entered as vectors:

        .. code-block:: python

            probabilities[m] = [P(0|m), P(1|m), ..., P(2 ** N - 1|m)]

        where ``P(j|m)`` is the probability of recording a measurement outcome
        of ``m`` as the value ``j``. Where ``j`` and ``m`` are integer
        representations of bit-strings.

        **Example: 1-qubit**

        .. code-block:: python

            probabilities[0] = [P("0"|"0"), P("1"|"0")]
            probabilities[1] = [P("0"|"1"), P("1"|"1")]

        **Example: 2-qubit**

        .. code-block:: python

            probabilities[0] = [P("00"|"00"), P("01"|"00"), P("10"|"00"), P("11"|"00")]
            probabilities[1] = [P("00"|"01"), P("01"|"01"), P("10"|"01"), P("11"|"01")]
            probabilities[2] = [P("00"|"10"), P("01"|"10"), P("10"|"10"), P("11"|"10")]
            probabilities[3] = [P("00"|"11"), P("01"|"11"), P("10"|"11"), P("11"|"11")]

        Args:
            probabilities (matrix): List of outcome assignment probabilities.
            atol (double): Threshold for checking probabilities are normalized
                           (Default: 1e-8).
        """
        self._check_probabilities(probabilities, atol)
        self._probabilities = np.array(probabilities, dtype=float)
        self._number_of_qubits = qubits_from_mat(probabilities)

    def __repr__(self):
        """Display ReadoutError."""
        return "ReadoutError({})".format(self._probabilities)

    def __str__(self):
        """Print error information."""
        output = "ReadoutError on {} qubits.".format(self._number_of_qubits) + \
                 " Assignment probabilities:"
        for j, vec in enumerate(self._probabilities):
            output += "\n P(j|{0}) =  {1}".format(j, vec)
        return output

    def __eq__(self, other):
        """Test if two ReadoutErrors are equal."""
        if not isinstance(other, ReadoutError):
            return False
        if self.number_of_qubits != other.number_of_qubits:
            return False
        return np.allclose(self._probabilities, other._probabilities,
                           atol=self.atol, rtol=self.rtol)

    def copy(self):
        """Make a copy of current ReadoutError."""
        # pylint: disable=no-value-for-parameter
        # The constructor of subclasses from raw data should be a copy
        return copy.deepcopy(self)

    @property
    def number_of_qubits(self):
        """Return the number of qubits for the error."""
        return self._number_of_qubits

    @property
    def probabilities(self):
        """Return the readout error probabilities matrix."""
        return self._probabilities

    @property
    def atol(self):
        """The default absolute tolerance parameter for float comparisons."""
        return ReadoutError._ATOL_DEFAULT

    @property
    def rtol(self):
        """The relative tolerance parameter for float comparisons."""
        return ReadoutError._RTOL_DEFAULT

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

    def ideal(self):
        """Return True if current error object is an identity"""
        iden = np.eye(2**self.number_of_qubits)
        delta = round(norm(np.array(self.probabilities) - iden), 12)
        if delta == 0:
            return True
        return False

    def to_instruction(self):
        """Convert the ReadoutError to a circuit Instruction."""
        return Instruction("roerror", 0, self.number_of_qubits, self._probabilities)

    def to_dict(self):
        """Return the current error as a dictionary."""
        error = {
            "type": "roerror",
            "operations": ["measure"],
            "probabilities": self._probabilities.tolist()
        }
        return error

    def compose(self, other, front=False):
        """Return the composition readout error other * self.

        Note that for `front=True` this is equivalent to the
        :meth:`ReadoutError.dot` method.

        Args:
            other (ReadoutError): a readout error.
            front (bool): If True return the reverse order composation
                          self * other instead [default: False].

        Returns:
            ReadoutError: The composition readout error.

        Raises:
            NoiseError: if other is not a ReadoutError or has incompatible
            dimensions.
        """
        if front:
            return self._matmul(other)
        return self._matmul(other, left_multiply=True)

    def dot(self, other):
        """Return the composition readout error self * other.

        Args:
            other (ReadoutError): a readout error.

        Returns:
            ReadoutError: The composition readout error.

        Raises:
            NoiseError: if other is not a ReadoutError or has incompatible
            dimensions.
        """
        return self._matmul(other)

    def power(self, n):
        """Return the compose of the readout error with itself n times.

        Args:
            n (int): the number of times to compose with self (n>0).

        Returns:
            ReadoutError: the n-times composition channel.

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
        """Return the tensor product readout error self ⊗ other.

        Args:
            other (ReadoutError): a readout error.

        Returns:
            ReadoutError: the tensor product readout error self ⊗ other.

        Raises:
            NoiseError: if other is not a ReadoutError.
        """
        return self._tensor_product(other, reverse=False)

    def expand(self, other):
        """Return the tensor product readout error self ⊗ other.

        Args:
            other (ReadoutError): a readout error.

        Returns:
            ReadoutError: the tensor product readout error other ⊗ self.

        Raises:
            NoiseError: if other is not a ReadoutError.
        """
        return self._tensor_product(other, reverse=True)

    @staticmethod
    def _check_probabilities(probabilities, threshold):
        """Check probabilities are valid."""
        # probabilities parameter can be a list or a numpy.ndarray
        if (isinstance(probabilities, list) and not probabilities) or \
           (isinstance(probabilities, np.ndarray) and probabilities.size == 0):
            raise NoiseError("Input probabilities: empty.")
        num_outcomes = len(probabilities[0])
        num_qubits = int(np.log2(num_outcomes))
        if 2**num_qubits != num_outcomes:
            raise NoiseError("Invalid probabilities: length "
                             "{} != 2**{}".format(num_outcomes, num_qubits))
        if len(probabilities) != num_outcomes:
            raise NoiseError("Invalid probabilities.")
        for vec in probabilities:
            arr = np.array(vec)
            if len(arr) != num_outcomes:
                raise NoiseError(
                    "Invalid probabilities: vectors are different lengths.")
            if abs(sum(arr) - 1) > threshold:
                raise NoiseError("Invalid probabilities: sum({})= {} "
                                 "is not 1.".format(vec, sum(arr)))
            if arr[arr < 0].size > 0:
                raise NoiseError(
                    "Invalid probabilities: {} "
                    "contains a negative probability.".format(vec))

    def _matmul(self, other, left_multiply=False):
        """Return the composition readout error.

        Args:
            other (ReadoutError): a readout error.
            left_multiply (bool): If True return other * self
                                  If False return self * other [Default:False]
        Returns:
            ReadoutError: The composition readout error.

        Raises:
            NoiseError: if other is not a ReadoutError or has incompatible
            dimensions.
        """
        if not isinstance(other, ReadoutError):
            other = ReadoutError(other)
        if self.number_of_qubits != other.number_of_qubits:
            raise NoiseError("other must have same number of qubits.")
        if left_multiply:
            probs = np.dot(other._probabilities, self._probabilities)
        else:
            probs = np.dot(self._probabilities, other._probabilities)
        return ReadoutError(probs)

    def _tensor_product(self, other, reverse=False):
        """Return the tensor product readout error.

        Args:
            other (ReadoutError): a readout error.
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False
        Returns:
            ReadoutError: the tensor product readout error.
        """
        if not isinstance(other, ReadoutError):
            other = ReadoutError(other)
        if reverse:
            probs = np.kron(other._probabilities, self._probabilities)
        else:
            probs = np.kron(self._probabilities, other._probabilities)
        return ReadoutError(probs)

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
            "'ReadoutError' does not support scalar multiplication.")

    def __truediv__(self, other):
        raise NotImplementedError("'ReadoutError' does not support division.")

    def __add__(self, other):
        raise NotImplementedError("'ReadoutError' does not support addition.")

    def __sub__(self, other):
        raise NotImplementedError(
            "'ReadoutError' does not support subtraction.")

    def __neg__(self):
        raise NotImplementedError("'ReadoutError' does not support negation.")
