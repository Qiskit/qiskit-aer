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
Aer Provider (:mod:`qiskit_aer`)
==========================================

.. currentmodule:: qiskit_aer

Simulator Provider
==================

.. autosummary::
    :toctree: ../stubs/

    AerProvider

Simulator Backends
==================

.. autosummary::
    :toctree: ../stubs/

    AerSimulator

Legacy Simulator Backends
=========================

.. autosummary::
    :toctree: ../stubs/

    QasmSimulator
    StatevectorSimulator
    UnitarySimulator

Exceptions
==========
.. autosummary::
   :toctree: ../stubs/

   AerError
"""

__lazy_modules__ = [
    "qiskit_aer.aerprovider",
    "qiskit_aer.jobs",
    "qiskit_aer.aererror",
    "qiskit_aer.backends",
    "qiskit_aer.library",
    "qiskit_aer.quantum_info",
    "qiskit_aer.noise",
    "qiskit_aer.utils",
    "qiskit_aer.version",
]

import platform

# https://github.com/Qiskit/qiskit-aer/issues/1
# Because of this issue, we need to make sure that Numpy's OpenMP library is initialized
# before loading our simulators, so we force it using this ugly trick
if platform.system() == "Darwin":
    import numpy as np

    np.dot(np.zeros(100), np.zeros(100))
# ... ¯\_(ツ)_/¯

# pylint: disable=wrong-import-position
# The following imports are "import X as X" rather tham "from ... import X"
# because the latter doesn't work for lazily importing module references into
# other modules (e.g. "library" into "qiskit_aer").
# pylint: disable=consider-using-from-import
import qiskit_aer.library as library
import qiskit_aer.quantum_info as quantum_info
import qiskit_aer.noise as noise
import qiskit_aer.utils as utils
# pylint: enable=consider-using-from-import
from qiskit_aer.aerprovider import AerProvider
from qiskit_aer.jobs import AerJob
from qiskit_aer.aererror import AerError
from qiskit_aer.backends import AerSimulator, QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit_aer.version import __version__


def __getattr__(name):
    # __getattr__ allows global Aer reference without triggering the resolution
    # of lazy imports
    if name == "Aer":
        aer = AerProvider()
        # Global instance to be used as the entry point for convenience.
        globals()["Aer"] = aer
        return aer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
