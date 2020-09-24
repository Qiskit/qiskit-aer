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

from abc import ABC, abstractmethod
from typing import Callable, Union, List
from .signals import BaseSignal, Signal, PiecewiseConstant

from numpy import convolve, array


class BaseTransferFunction(ABC):
    """
    Base class for transforming signals.
    """

    @abstractmethod
    def apply(self, signal: Union[BaseSignal, List[BaseSignal]]) -> Union[BaseSignal, List[BaseSignal]]:
        """
        Applies a transformation on a signal, such as a convolution,
        low pass filter, etc.

        Args:
            signal: A signal to which the transfer function will be applied.

        Returns:
            BaseSignal: The transformed signal.
        """
        raise NotImplementedError


class Convolution(BaseTransferFunction):
    """
    Applies a convolution as a sum

        (f*g)(n) = sum_k f(k)g(n-k)

    The implementation is quadratic in the number of samples in the signal.
    """

    def __init__(self, func: Callable):
        """
        Args:
            func: The convolution function specified in time. This function will be normalized
                  to one before doing the convolution. To scale signals multiply them by a float.
        """
        self._func = func

    def apply(self, signal: Union[Signal, List[Signal]]) -> Union[BaseSignal, List[BaseSignal]]:
        """
        Applies a transformation on a signal, such as a convolution,
        low pass filter, etc. Once a convolution is applied the signal
        can longer have a carrier as the carrier is part of the signal
        value and gets convolved.

        Args:
            signal: A signal or list of signals to which the transfer function will be applied.

        Returns:
            signal: The transformed signal or list of signals.
        """

        if isinstance(signal, List):
            convolved = []
            for sig in signal:
                convolved.append(self._convolve(sig))

            return convolved
        else:
            return self._convolve(signal)

    def _convolve(self, signal: BaseSignal) -> BaseSignal:
        """
        Helper function that applies the convolution to a single signal.

        Args:
            signal: The signal to convolve.

        Returns:
            signal: The transformed signal.
        """
        if isinstance(signal, PiecewiseConstant):
            # Perform a discrete time convolution.
            dt = signal.dt
            func_samples = array([self._func(dt*i) for i in range(signal.duration)])
            func_samples = func_samples / sum(func_samples)
            sig_samples = [signal.value(dt*i) for i in range(signal.duration)]

            convoluted_samples = convolve(func_samples, sig_samples)

            return PiecewiseConstant(dt, convoluted_samples, carrier_freq=0.)


class FFTConvolution(BaseTransferFunction):
    """
    Applies a convolution by moving into the fourier domain.
    """

    def __init__(self, func: Callable):
        self._func = func

    def apply(self, signal: Signal) -> Signal:
        raise NotImplementedError
