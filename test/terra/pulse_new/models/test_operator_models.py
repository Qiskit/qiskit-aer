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
"""tests for operator_models.py"""

import unittest
import numpy as np
from scipy.linalg import expm
from qiskit.providers.aer.pulse_new.models.operator_models import FrameFreqHelper, OperatorModel, vector_apply_diag_frame


class Test_FrameFreqHelper(unittest.TestCase):

    def setUp(self):
        pass
