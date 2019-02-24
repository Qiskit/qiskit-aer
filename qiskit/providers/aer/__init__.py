# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# https://github.com/Qiskit/qiskit-aer/issues/1
# Because of this issue, we need to make sure that Numpy's OpenMP library is initialized
# before loading our simulators, so we force it using this ugly trick
import platform
if platform.system() == "Darwin":
    import numpy as np
    np.dot(np.zeros(100), np.zeros(100))
# ... ¯\_(ツ)_/¯

"""Aer Backends."""

from .aerprovider import AerProvider
from .aerjob import AerJob
from .aererror import AerError
from .backends import *
from . import noise
from . import utils
from .version import __version__

# Global instance to be used as the entry point for convenience.
Aer = AerProvider()  # pylint: disable=invalid-name
