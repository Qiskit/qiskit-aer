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
==========================================
Aer Provider (:mod:`qiskit.providers.aer`)
==========================================

.. currentmodule:: qiskit.providers.aer

Simulator Provider
==================

.. autosummary::
    :toctree: ../stubs/

    AerProvider

Simulator Backends
==================

.. autosummary::
    :toctree: ../stubs/

    QasmSimulator
    StatevectorSimulator
    UnitarySimulator
    PulseSimulator

Job Class
=========
.. autosummary::
   :toctree: ../stubs/

   AerJob

Exceptions
==========
.. autosummary::
   :toctree: ../stubs/

   AerError
"""

# https://github.com/Qiskit/qiskit-aer/issues/1
# Because of this issue, we need to make sure that Numpy's OpenMP library is initialized
# before loading our simulators, so we force it using this ugly trick
import platform
if platform.system() == "Darwin":
    import numpy as np
    np.dot(np.zeros(100), np.zeros(100))
# ... ¯\_(ツ)_/¯

# pylint: disable=wrong-import-position
from .aerprovider import AerProvider
from .aerjob import AerJob
from .aererror import AerError
from .backends import *
from . import pulse
from . import noise
from . import utils
from .version import __version__

# Global instance to be used as the entry point for convenience.
Aer = AerProvider()  # pylint: disable=invalid-name
