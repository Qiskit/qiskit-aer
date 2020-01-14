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
Tests for option handling in digest.py
"""

import unittest
from test.terra.common import QiskitAerTestCase
#import qiskit
#import qiskit.pulse as pulse
#from qiskit.pulse import pulse_lib
#from qiskit.compiler import assemble
from qiskit.providers.aer.openpulse.qobj.digest import digest_pulse_obj


class TestDigest(QiskitAerTestCase):
    """Testing of functions in providers.aer.openpulse.qobj.digest."""
    def setUp(self):
        pass



if __name__ == '__main__':
    unittest.main()
