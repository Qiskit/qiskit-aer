# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import Callable, Union, List, Optional
import numpy as np

from .signals import VectorSignal, Signal
from qiskit.quantum_info.operators import Operator

class OperatorModel:
    """OperatorModel representing a sum of :class:`Operator` objects with
    time dependent coefficients.

    Specifically, this object represents a time dependent matrix of
    the form:

    .. math::

        G(t) = \sum_{i=0}^{k-1} s_i(t) G_i,

    for :math:`G_i` matrices (represented by :class:`Operator` objects),
    and the :math:`s_i(t)` given by signals represented by a
    :class:`VectorSignal` object, or a list of :class:`Signal` objects.

    This object contains functionality to evaluate :math:`G(t)` for a given
    :math:`t`, or to compute products :math:`G(t)A` and :math:`AG(t)` for
    :math:`A` an :class:`Operator` or array of suitable dimension.

    Additionally, this class has functionality for representing :math:`G(t)`
    in a rotating frame, and setting frequency cutoffs
    (e.g. for the rotating wave approximation).
    Specifically, given an anti-Hermitian frame operator :math:`F` (i.e.
    :math:`F^\dagger = -F`), entering the frame of :math:`F` results in
    the object representing the rotating operator :math:`e^{-Ft}G(t)e^{Ft} - F`.

    Further, if a frequency cutoff is set, when evaluating the
    `OperatorModel`, any terms with a frequency above the cutoff
    (which combines both signal frequency information and frame frequency
    information) will be set to :math:`0`.

    The signals in the model can be specified either directly (by giving a
    list of Signal objects or a VectorSignal), or by specifying a
    signal_mapping, defined as any function with return type
    `Union[List[Signal], VectorSignal]`. In this mode, assignments to the
    signal attribute will be treated as inputs to the signal_mapping. E.g.

    .. code-block:: python

        signal_map = lambda a: [Signal(lambda t: a * t, 1.)]
        model = OperatorModel(operators=[op], signal_mapping=signal_map)

        # setting signals now will pass the value into the signal_map function
        model.signals = 2.

        # the stored signals (retrivable with model.signals) is now
        # the output of signal_map(2.), converted to a VectorSignal

    See the signals property setter for a more detailed description.
    """

    def __init__(self,
                 operators: List[Operator],
                 signals: Optional[Union[VectorSignal, List[Signal]]] = None,
                 signal_mapping: Optional[Callable] = None,
                 frame_operator: Optional[Union[Operator, np.array]] = None,
                 cutoff_freq: Optional[float] = None):
        """Initialize.

        Args:
            operators: list of Operator objects.
            signals: Specifiable as either a VectorSignal, a list of
                     Signal objects, or as the inputs to signal_mapping.
                     OperatorModel can be instantiated without specifying
                     signals, but it can not perform any actions without them.
            signal_mapping: a function returning either a
                            VectorSignal or a list of Signal objects.

            frame_operator: Rotating frame operator. If specified with a 1d
                            array, it is interpreted as the diagonal of a
                            diagonal matrix.
            cutoff_freq: Frequency cutoff when evaluating the model.
        """

        self._operators = operators

        self._frame_operator = frame_operator
        self._cutoff_freq = cutoff_freq

        # initialize signal-related attributes
        self._signal_params = None
        self._signals = None
        self._carrier_freqs = None
        self.signal_mapping = signal_mapping

        if signals is not None:
            # note: setting signals includes a call to _construct_frame_helper()
            self.signals = signals
        else:
            self._construct_frame_helper()

    @property
    def signals(self) -> VectorSignal:
        """Return the signals in the model."""
        return self._signals

    @signals.setter
    def signals(self, signals: Union[VectorSignal, List[Signal]]):
        """Set the signals.

        Behavior:
            - If no signal_mapping is specified, the argument signals is
              assumed to be either a list of Signal objects, or a VectorSignal,
              and is saved in self._signals.
            - If a signal_mapping is specified, signals is assumed to be a valid
              input of signal_mapping. The argument signals is set to
              self._signal_params, and the output of signal_mapping is saved in
              self._signals.
        """
        if signals is None:
            self._signal_params = None
            self._signals = None
            self._carrier_freqs = None
        else:

            # if a signal_mapping is specified, take signals as the input
            if self.signal_mapping is not None:
                self._signal_params = signals
                signals = self.signal_mapping(signals)

            # if signals is a list, instantiate a VectorSignal
            if isinstance(signals, list):
                signals = VectorSignal.from_signal_list(signals)

            # if it isn't a VectorSignal by now, raise an error
            if not isinstance(signals, VectorSignal):
                raise Exception('signals specified in unaccepted format.')

            # verify signal length is same as operators
            if len(signals.carrier_freqs) != len(self._operators):
                raise Exception("""signals needs to have the same length as
                                    operators.""")

            # check if the new carrier frequencies are different from the old.
            # if yes, update them and reinstantiate the frame helper.
            new_freqs = signals.carrier_freqs
            if any(signals.carrier_freqs != self._carrier_freqs):
                self._carrier_freqs = signals.carrier_freqs
                self._construct_frame_helper()

            self._signals = signals

    @property
    def frame_operator(self) -> Union[Operator, np.array]:
        """Return the frame operator."""
        return self._frame_operator

    @frame_operator.setter
    def frame_operator(self, frame_operator: Union[Operator, np.array]):
        """Set the frame operator; must be anti-Hermitian. See class
        docstring for the effects of setting a frame.

        Accepts frame_operator as:
            - An Operator object
            - A 2d array
            - A 1d array - in which case it is assumed the frame operator is a
              diagonal matrix, and the array gives the diagonal elements.
        """

        if isinstance(frame_operator, np.ndarray) and frame_operator.ndim == 1:
            # if 1d array check purely imaginary
            if np.linalg.norm(frame_operator + frame_operator.conj()) > 10**-10:
                raise Exception("""frame_operator must be an
                                   anti-Hermitian matrix.""")
        else:
            # otherwise, cast as Operator and verify anti-Hermitian
            frame_operator = Operator(frame_operator)

            herm_part = frame_operator + frame_operator.adjoint()
            if herm_part != Operator(np.zeros(frame_operator.dim)):
                raise Exception("""frame_operator must be an
                                   anti-Hermitian matrix.""")

        self._frame_operator = frame_operator
        self._construct_frame_helper()

    @property
    def cutoff_freq(self) -> float:
        """Return the cutoff frequency."""
        return self._cutoff_freq

    @cutoff_freq.setter
    def cutoff_freq(self, cutoff_freq: float):
        """Set the cutoff frequency."""
        if cutoff_freq != self._cutoff_freq:
            self._cutoff_freq = cutoff_freq
            self._construct_frame_helper()

    def evaluate(self, time: float, in_frame_basis: bool = False) -> np.array:
        """
        Evaluate the model in array format.

        Args:
            time: Time to evaluate the model
            in_frame_basis: Whether to evaluate in the basis in which the frame
                            operator is diagonal

        Returns:
            np.array: the evaluated model
        """

        if self.signals is None:
            raise Exception("""OperatorModel cannot be
                               evaluated without signals.""")

        sig_envelope_vals = self.signals.envelope_value(time)

        return self._frame_freq_helper.evaluate(time,
                                                sig_envelope_vals,
                                                in_frame_basis)

    def lmult(self,
              time: float,
              y: np.array,
              in_frame_basis: bool = False) -> np.array:
        """
        Return the product evaluate(t) @ y.

        Args:
            time: Time at which to create the generator.
            y: operator or vector to apply the model to.
            in_frame_basis: whether to evaluate in the frame basis

        Returns:
            np.array: the product
        """
        return np.dot(self.evaluate(time, in_frame_basis), y)

    def rmult(self,
              time: float,
              y: np.array,
              in_frame_basis: bool = False) -> np.array:
        """
        Return the product y @ evaluate(t).

        Args:
            time: Time at which to create the generator.
            y: operator or vector to apply the model to.
            in_frame_basis: whether to evaluate in the frame basis

        Returns:
            np.array: the product
        """
        return np.dot(y, self.evaluate(time, in_frame_basis))

    @property
    def drift(self) -> np.array:
        """Return the part of the model with only Constant coefficients as a
        numpy array.
        """

        # for now if the frame operator is not None raise an error
        if self.frame_operator is not None:
            raise Exception("""The drift is currently ill-defined if
                               frame_operator is not None.""")

        drift_env_vals = self.signals.drift_array

        return self._frame_freq_helper.evaluate(0, drift_env_vals)

    def _construct_frame_helper(self):
        """Helper function for constructing frame helper from relevant
        attributes.
        """
        self._frame_freq_helper = FrameFreqHelper(self._operators,
                                                  self._carrier_freqs,
                                                  self.frame_operator,
                                                  self.cutoff_freq)


class FrameFreqHelper:
    """A helper class containing some technical calculations for evaluating
    an operator model in a rotating frame, potentially with a cutoff frequency.

    Specifically, the frame operator and all operators are converted into a
    basis in which the frame operator is diagonal (which we call the
    'frame basis' in comments), and all calculations are
    done in that basis to save on the cost of evaluating exp(Ft) and exp(-Ft).

    It also contains additional helper methods for converting a 'state' into
    and out of the frame, and into/out of the frame basis, which are useful
    if the operator model is being used as the generator G(t) in a DE of the
    form y'(t) = G(t)y(t).

    Evaluation and frame conversion methods for a 'state' all have additional
    options to specify which basis (the original or frame basis) the
    input/output is specified in. E.g., when solving the DE y'(t) = G(t)y(t)
    in some frame F, it is convenient to solve the DE fully in the basis in
    which F is diagonal (avoiding basis change operations when evaluating
    G(t)), and only convert back to the original basis when necessary.
    """

    def __init__(self,
                 operators: List[Operator],
                 carrier_freqs: Optional[np.array] = None,
                 frame_operator: Optional[Union[Operator, np.array]] = None,
                 cutoff_freq: Optional[float] = None):
        """
        Initialize.

        Args:
            operators: List of Operator objects.
            carrier_freqs: List of carrier frequencies for the
                           coefficients of the operators. If None, defaults to
                           the zero vector.
            frame_operator: frame operator which must be anti-Hermitian. If
                            given as a 1d array, assumed to be the diagonal
                            of a diagonal matrix. If None, defaults to the 0
                            matrix.
            cutoff_freq: Cutoff frequency when evaluating generator.
                         If None, no cutoff is performed.
        """

        # if None, set to a 1d array of zeros
        if frame_operator is None:
            frame_operator = np.zeros(operators[0].dim[0])

        # if frame_operator is a 1d array, assume already diagonalized
        if isinstance(frame_operator, np.ndarray) and frame_operator.ndim == 1:

            # verify that it is anti-hermitian (i.e. purely imaginary)
            if np.linalg.norm(frame_operator + frame_operator.conj()) > 10**-10:
                raise Exception("""frame_operator must be an
                                   anti-Hermitian matrix.""")

            self.frame_diag = frame_operator
            self.frame_basis = np.eye(len(frame_operator))
            self.frame_basis_adjoint = self.frame_basis
        # if not, diagonalize it
        else:
            # Ensure that it is an Operator object
            frame_operator = Operator(frame_operator)

            # verify anti-hermitian
            herm_part = frame_operator + frame_operator.adjoint()
            if herm_part != Operator(np.zeros(frame_operator.dim)):
                raise Exception("""frame_operator must be an
                                   anti-Hermitian matrix.""")

            # diagonalize with eigh, utilizing assumption of anti-hermiticity
            frame_diag, frame_basis = np.linalg.eigh(1j * frame_operator.data)

            self.frame_diag = -1j * frame_diag
            self.frame_basis = frame_basis
            self.frame_basis_adjoint = frame_basis.conj().transpose()

        # rotate operators into frame_basis and store as a 3d array
        self._operators_in_frame_basis = np.array([self.frame_basis_adjoint @
                                                   op.data @
                                                   self.frame_basis for
                                                   op in operators])

        # set up carrier frequencies and cutoff
        if carrier_freqs is None:
            carrier_freqs = np.zeros(len(operators))
        self.carrier_freqs = carrier_freqs

        self.cutoff_freq = cutoff_freq

        # create difference matrix for diagonal elements
        dim = len(self.frame_diag)
        D_diff = np.ones((dim, dim)) * self.frame_diag
        D_diff = D_diff - D_diff.transpose()

        # set up matrix encoding frequencies
        im_angular_freqs = 1j * 2 * np.pi * self.carrier_freqs
        self._freq_array = np.array([w + D_diff for w in im_angular_freqs])

        # set up frequency cutoff matrix - i.e. same shape as self._S - with
        # each entry a 1 if the corresponding entry of self._S has a frequency
        # below the cutoff, and 0 otherwise
        self._cutoff_array = None
        if cutoff_freq is not None:
            self._cutoff_array = ((np.abs(self._freq_array.imag) / (2 * np.pi))
                                    < self.cutoff_freq).astype(int)

    def evaluate(self,
                 t: float,
                 coefficients: np.array,
                 in_frame_basis: bool = False) -> np.array:
        """Evaluate the operator in the frame at a given time, for a given
        array of coefficients.

        Args:
            t: time
            coefficients: coefficients for each operator
            in_frame_basis: whether to return in the basis in which
                            the frame operator is diagonal or not

        Returns:
            np.array the evaluated operator
        """

        # first evaluate the unconjugated coefficients for each matrix element,
        # given by the coefficient for the full matrix multiplied by the
        # exponentiated frequency term for each entry
        Q = (coefficients[:, np.newaxis, np.newaxis] *
                np.exp(self._freq_array * t))

        # apply cutoff if present
        if self._cutoff_array is not None:
            Q = self._cutoff_array * Q

        # multiplying the operators by the average of the "unconjugated" and
        # "conjugated" coefficients
        op_list = (0.5 * (Q + Q.conj().transpose(0, 2, 1)) *
                   self._operators_in_frame_basis)

        # sum the operators and subtract the frame operator
        op_in_frame_basis = np.sum(op_list, axis=0) - np.diag(self.frame_diag)

        if in_frame_basis:
            return op_in_frame_basis
        else:
            return (self.frame_basis @ op_in_frame_basis @
                    self.frame_basis_adjoint)

    def state_into_frame(self,
                         t: float,
                         y: np.array,
                         y_in_frame_basis: bool = False,
                         return_in_frame_basis: bool = False):
        """Take a state into the frame, i.e. return exp(-Ft) @ y.

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the frame basis
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """

        out = y

        # if not in frame basis convert it
        if not y_in_frame_basis:
            out = self.state_into_frame_basis(out)

        out = np.diag(np.exp(- t * self.frame_diag)) @ out

        if not return_in_frame_basis:
            out = self.state_out_of_frame_basis(out)

        return out

    def state_out_of_frame(self,
                           t: float,
                           y: np.array,
                           y_in_frame_basis: bool = False,
                           return_in_frame_basis: bool = False):
        """Bring a state out of the frame, i.e. return exp(Ft) @ y.

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the frame basis
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """
        # same calculation as state_into_frame, just with -time
        return self.state_into_frame(-t,
                                     y,
                                     y_in_frame_basis,
                                     return_in_frame_basis)

    def state_into_frame_basis(self, y: np.array):
        return self.frame_basis_adjoint @ y

    def state_out_of_frame_basis(self, y: np.array):
        return self.frame_basis @ y
