# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Noise module for Qiskit Aer.

This module contains classes and functions to build a noise model for
simulating a Qiskit quantum circuit in the presence of errors.

The main noise model class is `NoiseModel` class. Errors are represented
using the `QuantumError` and `ReadoutError` classes and can be added to a
noise model to sample errors during specific circuit operations of a
simulation.

noise  (package)
|
|-- Noise Model (class)
|-- QuantumError (class)
|-- ReadoutError (class)
|-- errors (package)
"""

from .noise_model import NoiseModel
from .quantum_error import QuantumError
from .readout_error import ReadoutError
from . import errors
