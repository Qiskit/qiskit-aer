# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Aer Backends."""

from .aerprovider import AerProvider
from .aerjob import AerJob
from .backends import *
#from .backends.qasm_simulator import QasmSimulator
#from .backends.statevector_simulator import StatevectorSimulator
#from .backends.unitary_simulator import UnitarySimulator
from .noise import *
from .utils import *
from .version import __version__

# Global instance to be used as the entry point for convenience.
Aer = AerProvider()  # pylint: disable=invalid-name
