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
===========================================================
Circuit Extensions (:mod:`qiskit.providers.aer.extensions`)
===========================================================

.. currentmodule:: qiskit.providers.aer.extensions

Snapshots
=========

.. note:

    Snapshot extensions will be deprecated after qiskit-aer 0.8 release.
    They have been superceded by the save instructions in the
    :mod:`qiskit.providers.aer.library` module.

Snapshot instructions allow taking a snapshot of the current state of the
simulator without effecting the outcome of the simulation. These can be
used with the `QasmSimulator` backend to return the expectation value of
an operator or the probability of measurement outcomes.

.. autosummary::
    :toctree: ../stubs/

    Snapshot
    SnapshotProbabilities
    SnapshotExpectationValue
    SnapshotStatevector
    SnapshotDensityMatrix
    SnapshotStabilizer
"""

from .snapshot import *
from .snapshot_statevector import *
from .snapshot_stabilizer import *
from .snapshot_density_matrix import *
from .snapshot_probabilities import *
from .snapshot_expectation_value import *
