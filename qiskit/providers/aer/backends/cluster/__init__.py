# -*- coding: utf-8 -*-

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
Cluster Backend (:mod:`qiskit.providers.aer.backends.cluster`)
===========================================================

.. currentmodule:: qiskit.providers.aer.backends.cluster

High level mechanism for handling cluster jobs.

Classes
==========================
.. autosummary::
   :toctree: ../stubs/

   ClusterBackend
"""

from .cluster_backend import ClusterBackend
from .clusterjobset import JobSet
from .clusterjob import CJob
from .clusterresults import CResults
from .exceptions import *
