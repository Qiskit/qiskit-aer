# -*- coding: utf-8 -*-

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

from signals import Signal, Constant
from transfer_functions import BaseTransferFunction
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators import Operator

class BaseModel:

    def __init__(self,
                 signals: List[Signal],
                 operators: List[BaseOperator],
                 transfer_functions: List[List[BaseTransferFunction]] = None,
                 transformations=None):
        """
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

        if isinstance(transformations, dict):
            self.frame_operator = transformations.get('frame')
            self.rwa_freq_cutoff = transformations.get('rwa_freq_cutoff')
        else:
            self.frame_operator = None
            self.rwa_freq_cutoff = None

        # initialize signals
        self._signals = None
        self._carrier_freqs = None

        if signals is not None:
            # note: setting signals includes a call to enter_frame
            self.signals = signals
        else:
            self.enter_frame(self.frame_operator, self.rwa_freq_cutoff)

        if transfer_functions is not None:
            if len(signals) != len(transfer_functions):
                raise
        else:
            self._transfer_functions = None


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
                self.enter_frame(self.frame_operator, self.rwa_freq_cutoff)

    def evaluate(self, time: float, in_frame_diag_basis: bool = False) -> np.array:
        """
        Return the generator of the model in matrix format

        Args:
            time: Time to evaluate the model
            in_frame_diag_basis: Whether or not to evaluate in the basis in which the frame
                                 operator is diagonal
        """

        sig_envelope_vals = np.array([sig.envelope_value(time) for sig in self.signals])

        return self._model_frame_signal_helper.generator_in_frame(time,
                                                                  sig_envelope_vals,
                                                                  in_frame_diag_basis)

    def lmult(self, time: float, y: np.array, in_frame_diag_basis: bool = False) -> np.array:
        """
        Return the dot product generator(t) * y.

        Args:
            time: Time at which to create the generator.
            y: operator or vector to apply the model to.
            in_frame_diag_basis: whether to evaluate the frame in the frame basis
        """
        generator = self.evaluate(time, in_frame_diag_basis)

        return np.dot(generator, y)

    def rmult(self, time: float, y: np.array, in_frame_diag_basis: bool = False) -> np.array:
        generator = self.evaluate(time, in_frame_diag_basis)
        return np.dot(y, generator)

    def enter_frame(self, frame_operator=None, rwa_freq_cutoff=None):
        """Enters frame given by frame_operator potentially with rwa cutoff."""
        self.frame_operator = frame_operator
        self.rwa_freq_cutoff = rwa_freq_cutoff

        self._model_frame_signal_helper = ModelFrameSignalHelper(self._operators,
                                                                 frame_operator,
                                                                 self._carrier_freqs,
                                                                 rwa_freq_cutoff)

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


"""
This class is meant to help with two things:
- Working with a model in a rotating frame
- Handling carrier frequencies of signals
Both of these things need to be dealt with together when making the RWA:
- The steps necessary to make the rotating wave approximation involve a combination of
 frame information and channel frequency information

I am imagining this class to be:
- A helper class used by a model for handling/compartmentalizing the computations related to the
  above two pieces
- It should be helpful:
    - From the perspective of the model class, e.g. "enter a frame" and be able to evaluate the
      generator in that frame
    - From the perspective of the solvers - functionality for directly working in the basis
      in which the frame is diagonal (functions for getting the generator in that basis)

behavior:
- attributes
    - frame_operator
    - _frame_basis (basis in which frame_operator is diagonal)
    - _frame_basis_adj (adjoint of _frame_basis)
    - _frame_diag (diagonal of diagonalized frame_operator)
    - _operators_in_frame_basis (copy of operators in the basis _frame_basis)
    - _signal_freqs (frequencies of signals)
    - _S, _M (matrices for doing computations)
- initiall required methods
    - constructor - diagonalizes frame_operator, and sets everything up
    - generator_in_frame(t, signal_vals, in_frame_basis)


Note:
- for now we will store everything internally with numpy arrays, but maybe should move to pure
Operator usage - needs DiagonalOperator
"""
class ModelFrameSignalHelper:

    def __init__(self, operators, frame_operator=None, signal_freqs=None, rwa_freq_cutoff=None):
        """
        Set stuff up - if signal_freqs is None, take them all to be 0.

        Assume frame_op is anti-hermitian
        """

        # if None, initialize as the zero diagonal operator
        if frame_operator is None:
            frame_operator = np.zeros(operators[0].dim[0])

        # diagonalize frame

        if isinstance(frame_operator, np.ndarray) and frame_operator.ndim == 1:
            # set up if frame_op is already set as the diagonal of the operator

            # check anti-hermitian
            if np.linalg.norm(frame_operator + frame_operator.conj()) > 10**-10:
                raise Exception('frame_op must correspond to an anti-Hermitian matrix.')

            self._frame_diag = frame_operator
            self._frame_basis = np.eye(len(frame_operator))
            self._frame_basis_adjoint = self._frame_basis
        else:
            # should add diagonal operator
            frame_operator = Operator(frame_operator)

            # verify anti-hermitian
            herm_part = frame_operator + frame_operator.adjoint()
            if herm_part != Operator(np.zeros(frame_operator.dim)):
                raise Exception('frame_op must be an anti-Hermitian matrix.')

            # diagonalize with eigh, utilizing assumption of anti-hermiticity
            frame_diag, frame_basis = np.linalg.eigh(1j * frame_operator.data)

            self._frame_diag = -1j * frame_diag
            self._frame_basis = frame_basis
            self._frame_basis_adjoint = frame_basis.conj().transpose()

        # rotate operators into frame_basis
        self._operators_in_frame_basis = np.array([self._frame_basis_adjoint @ op.data @ self._frame_basis for op in operators])

        # set up signal frequencies
        if signal_freqs is None:
            signal_freqs = np.zeros(len(operators))

        self._signal_freqs = signal_freqs

        # set up helper matrices

        self._rwa_freq_cutoff = rwa_freq_cutoff

        # create difference matrix for diagonal elements
        dim = len(self._frame_diag)
        D_diff = np.ones((dim, dim)) * self._frame_diag #* np.ones((dim, 1))
        D_diff = D_diff - D_diff.transpose()

        # set up matrix encoding frequencies
        im_angular_freqs = 1j * 2 * np.pi * self._signal_freqs
        self._S = np.array([w + D_diff for w in im_angular_freqs])

        self._M_cutoff = None
        if rwa_freq_cutoff is not None:
            self._M_cutoff = ((np.abs(self._S.imag) / (2 * np.pi)) <
                                            self._rwa_freq_cutoff).astype(int)

    def generator_in_frame(self, t, signal_vals, in_frame_diag_basis=False):
        """Return the generator in the frame.
        """
        # get operators in diagonal frame with signal coefficients applied
        op_list = vector_apply_diag_frame(t,
                                          self._operators_in_frame_basis,
                                          signal_vals,
                                          self._S,
                                          self._M_cutoff)
        # generator in diagonal frame_basis
        gen_in_frame_diag = np.sum(op_list, axis=0) - np.diag(self._frame_diag)

        if in_frame_diag_basis:
            return gen_in_frame_diag
        else:
            return self._frame_basis @ gen_in_frame_diag @ self._frame_basis_adjoint

    def state_into_frame(self, t, y, in_frame_diag_basis=False):
        """Take a state into the frame."""

        y_in_diag_basis = np.exp(- t * self._frame_diag) * (self._frame_basis_adjoint @ y)

        if in_frame_diag_basis:
            return y_in_diag_basis
        else:
            return self._frame_basis @ y_in_diag_basis

    def state_out_of_frame(self, t, y, in_frame_diag_basis=False):
        """Bring a state out of the frame."""

        if in_frame_diag_basis:
            return self._frame_basis @ (np.exp(t * self._frame_diag) * y)
        else:
            return self._frame_basis @ np.diag(np.exp(t * self._frame_diag)) @ self._frame_basis_adjoint @ y



def vector_apply_diag_frame(t, mats_in_frame_basis, coeffs, S, M_cutoff=None):
    """
    Vectorized application of rotating frame with cutoffs
    """

    # entrywise exponential of S * t, and multiply each coeff by the corresponding
    # matrix in the 3d array
    Q = coeffs[:, np.newaxis, np.newaxis] * np.exp(S * t)

    if M_cutoff is not None:
        Q = M_cutoff * Q

    return 0.5 * (Q + Q.conj().transpose(0, 2, 1)) * mats_in_frame_basis
