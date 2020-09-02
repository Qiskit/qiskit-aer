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
from signals import Signal

class BaseTransferFunction(ABC):
    """
    Base class for transforming signals.
    """

    @abstractmethod
    def apply(self, signal: Signal) -> Signal:
        """
        Applies a transformation on a signal, such as a convolution,
        low pass filter, etc.

        Returns:
            BaseSignal: The transformed signal.
        """
        raise NotImplementedError
