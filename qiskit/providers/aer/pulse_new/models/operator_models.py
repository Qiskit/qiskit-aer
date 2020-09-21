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

from typing import List
import numpy as np

from .signals import Signal, Constant
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators import Operator

class OperatorModel:
    """OperatorModel representing a sum of :class:`Operator` with
    time dependent coefficients.

    Specifically, this object represents a time dependent matrix of
    the form:

    .. math::

        G(t) = \sum_{i=0}^{k-1} s_i(t) G_i,

    for :math:`G_i` matrices (represented by :class:`Operator` objects),
    and the :math:`s_i(t)` given by signals represented by a
    class:`VectorSignal` object. (remains to be seen)

    This object contains functionality to evaluate :math:`G(t)` for a given
    :math:`t`, or to compute products :math:`G(t)A` and :math:`AG(t)` for
    :math:`A` an :class:`Operator` or array of suitable dimension.

    Additionally, this class has functionality for representing :math:`G(t)`
    in a rotating frame, and doing a rotating wave approximation in that frame.
    Specifically, given an anti-Hermitian frame operator :math:`F` (i.e.
    :math:`F^\dagger = -F`), entering the frame of :math:`F` results in
    the object representing the operator :math:`e^{-Ft}G(t)e^{Ft} - F`.

    Further, if an RWA frequency cutoff is set, when evaluating the
    `OperatorModel`, any terms with a frequency above the cutoff
    (which combines both signal frequency information and frame frequency
    information) will be set to :math:`0`.
    """

    def __init__(self,
                 operators,
                 signals,
                 signal_mapping,
                 frame_operator=None,
                 cutoff_freq=None):
        """
        *** Fix this***!!!!!!!!!
        Currently:
        - operators are a list of Operator objects
        - signals are a list of Signal objects, but this should be changed to
          be either a VectorSignal or literally anything (the inputs to
          signal_mapping)
        - frame_operator (for now an Operator or an np array)
        - cutoff_freq (float)

        Args:
            signals: The signals of the model. Each signal is the coefficient (potentially
                time-dependent) to an operator. There should be as many signals in the model
                as there are operators.
            operators: The list of operators.
            transfer_functions: List of list of transfer functions applied to the signals.
                The outer list must have the same length as the list of signals.
            transformations: specification of transformations on the model
        """

        self._operators = operators

        self.frame_operator = frame_operator
        self.cutoff_freq = cutoff_freq

        # initialize signals
        self._signals = None
        self._carrier_freqs = None

        if signals is not None:
            # note: setting signals includes a call to enter_frame
            self.signals = signals
        else:
            self.enter_frame(self.frame_operator, self.cutoff_freq)

        """
        To do: add in handling of signal_mapping or whatever it ends up being
        called
        """

        self._signal_mapping = None


    """
    To do: update signal handling
    """

    @property
    def signals(self) -> List[Signal]:
        """Return the signals in the model"""
        return self._signals

    @signals.setter
    def signals(self, signals: List[Signal]):
        """Set the signals"""
        if signals is None:
            self._signals = None
            self._carrier_freqs = None
        else:
            if len(signals) != len(self._operators):
                raise

            self._signals = signals
            new_freqs = np.array([sig.carrier_freq for sig in signals])

            # if the new frequencies are different, recompile the frame/signal
            # information
            if any(new_freqs != self._carrier_freqs):
                self._carrier_freqs = new_freqs
                self.enter_frame(self.frame_operator, self.cutoff_freq)

    def evaluate(self, time: float, in_frame_diag_basis: bool = False) -> np.array:
        """
        Return the generator of the model in matrix format

        Args:
            time: Time to evaluate the model
            in_frame_diag_basis: Whether or not to evaluate in the basis in which the frame
                                 operator is diagonal
        """

        sig_envelope_vals = np.array([sig.envelope_value(time) for sig in self.signals])

        return self._frame_freq_helper.generator_in_frame(time,
                                                          sig_envelope_vals,
                                                          in_frame_diag_basis)

    def lmult(self, time: float, y: np.array, in_frame_diag_basis: bool = False) -> np.array:
        """
        Return the product evaluate(t) @ y.

        Args:
            time: Time at which to create the generator.
            y: operator or vector to apply the model to.
            in_frame_diag_basis: whether to evaluate the frame in the frame basis
        """
        return np.dot(self.evaluate(time, in_frame_diag_basis), y)

    def rmult(self, time: float, y: np.array, in_frame_diag_basis: bool = False) -> np.array:
        """
        Return the product y @ evaluate(t).

        Args:
            time: Time at which to create the generator.
            y: operator or vector to apply the model to.
            in_frame_diag_basis: whether to evaluate the frame in the frame basis
        """
        return np.dot(y, self.evaluate(time, in_frame_diag_basis))

    """
    To do: maybe make frame_operator and cutoff_freq properties, each of which
    can individually be changed
    """

    def enter_frame(self, frame_operator=None, cutoff_freq=None):
        """Enters frame given by frame_operator potentially with rwa cutoff.

        Note: this will undo any existing frame transformations
        """
        self.frame_operator = frame_operator
        self.cutoff_freq = cutoff_freq

        self._frame_freq_helper = FrameFreqHelper(self._operators,
                                                  frame_operator,
                                                  self._carrier_freqs,
                                                  cutoff_freq)

    """Make this a property?
    """

    def drift(self):
        """Return the part of the model with only Constant coefficients as a numpy array."""

        # for now if the frame operator is not None raise an error
        if self.frame_operator is not None:
            raise Exception('For now, the drift is ill-defined if the frame_operator is not None.')

        drift = np.zeros_like(self._operators[0].data)

        for sig, op in zip(self.signals, self._operators):
            if isinstance(sig, Constant):
                drift += sig.value() * op.data

        return drift


class FrameFreqHelper:
    """Contains some technical calculations for evaluating an operator model
    in a rotating frame, potentially with a cutoff frequency.
    """

    def __init__(self,
                 operators,
                 carrier_freqs=None,
                 frame_operator=None,
                 cutoff_freq=None):
        """
        Initialize.

        Args:
            operators (list): List of Operator objects.
            carrier_freqs (array): List of carrier frequencies for the
                                   coefficients of the operators.
            frame_operator (Operator): frame operator - either an Operator
                                       object or a 1d array (in which case it
                                       is assumed to already be diagonalized.)

                                       the frame_operator is assumed to be
                                       anti-hermitian.
            cutoff_freq (float): Cutoff frequency when evaluating generator.
                                 If None, no cutoff is performed.
        """

        # initial setup of frame operator

        # if None, set to a 1d array of zeros
        if frame_operator is None:
            frame_operator = np.zeros(operators[0].dim[0])

        # if frame_operator is a 1d array, assume already diagonalized
        if isinstance(frame_operator, np.ndarray) and frame_operator.ndim == 1:

            # verify that it is anti-hermitian (i.e. purely imaginary)
            if np.linalg.norm(frame_operator + frame_operator.conj()) > 10**-10:
                raise Exception('frame_op must correspond to an anti-Hermitian matrix.')

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
                raise Exception('frame_op must be an anti-Hermitian matrix.')

            # diagonalize with eigh, utilizing assumption of anti-hermiticity
            frame_diag, frame_basis = np.linalg.eigh(1j * frame_operator.data)

            self.frame_diag = -1j * frame_diag
            self.frame_basis = frame_basis
            self.frame_basis_adjoint = frame_basis.conj().transpose()

        # rotate operators into frame_basis
        self._operators_in_frame_basis = np.array([self.frame_basis_adjoint @ op.data @ self.frame_basis for op in operators])

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
        self._S = np.array([w + D_diff for w in im_angular_freqs])

        self._M_cutoff = None
        if cutoff_freq is not None:
            self._M_cutoff = ((np.abs(self._S.imag) / (2 * np.pi)) <
                                            self.cutoff_freq).astype(int)

    def evaluate(self, t, coefficients, in_frame_basis=False):
        """Evaluate the operator in the frame at a given time, for a given
        array of coefficients for each operator.

        Args:
            t (float): time
            coefficients (array): coefficients for each operator
            in_frame_basis (bool): whether to return in the basis in which
                                        the frame operator is diagonal or not

        Returns:
            array
        """
        # get operators in diagonal frame with signal coefficients applied
        op_list = vector_apply_diag_frame(t,
                                          self._operators_in_frame_basis,
                                          coefficients,
                                          self._S,
                                          self._M_cutoff)
        # generator in diagonal frame_basis
        op_in_frame_basis = np.sum(op_list, axis=0) - np.diag(self.frame_diag)

        if in_frame_basis:
            return op_in_frame_basis
        else:
            return self.frame_basis @ op_in_frame_basis @ self.frame_basis_adjoint

    def state_into_frame(self, t, y, y_in_frame_basis=False,
                                     return_in_frame_basis=False):
        """Take a state into the frame, i.e. return exp(-Ft) @ y.

        Args:
            t (float): time
            y (array): state (array of appropriate size)
            y_in_frame_basis (bool): whether or not the array y is already in
                                     the frame basis
            return_in_frame_basis (bool): whether or not to return the result
                                          in the frame basis
        """

        out_in_fb = None
        if y_in_frame_basis:
            out_in_fb = np.exp(- t * self.frame_diag) * y
        else:
            out_in_fb = (np.exp(- t * self.frame_diag) *
                               (self.frame_basis_adjoint @ y))

        if return_in_frame_basis:
            return out_in_fb
        else:
            return self.frame_basis @ out_in_fb

    def state_out_of_frame(self, t, y, y_in_frame_basis=False,
                                       return_in_frame_basis=False):
        """Bring a state out of the frame, i.e. return exp(Ft) @ y.

        Args:
            t (float): time
            y (array): state (array of appropriate size)
            y_in_frame_basis (bool): whether or not the array y is already in
                                     the frame basis
            return_in_frame_basis (bool): whether or not to return the result
                                          in the frame basis
        """

        out_in_fb = None
        if y_in_frame_basis:
            out_in_fb = np.exp(t * self.frame_diag) * y
        else:
            out_in_fb = np.diag(np.exp(t * self.frame_diag)) @ self.frame_basis_adjoint @ y

        if return_in_frame_basis:
            return out_in_fb
        else:
            return self.frame_basis @ out_in_fb


def vector_apply_diag_frame(t, mats_in_frame_basis, coeffs, S, M_cutoff=None):
    """Given a list of matrices specified in the frame_basis for a
    """

    # entrywise exponential of S * t, and multiply each coeff by the
    # corresponding matrix in the 3d array
    Q = coeffs[:, np.newaxis, np.newaxis] * np.exp(S * t)

    if M_cutoff is not None:
        Q = M_cutoff * Q

    return 0.5 * (Q + Q.conj().transpose(0, 2, 1)) * mats_in_frame_basis
