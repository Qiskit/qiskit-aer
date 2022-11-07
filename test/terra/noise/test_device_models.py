# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests for utility functions to create device noise model.
"""

from test.terra.common import QiskitAerTestCase

from qiskit.providers.fake_provider import FakeManilaV2
from qiskit_aer.noise.device.models import basic_device_gate_errors


class TestDeviceNoiseModel(QiskitAerTestCase):
    """Testing device noise model"""

    def test_basic_device_gate_errors_from_target(self):
        """Test if the resulting gate errors does not include errors on non-gate instructions"""
        target = FakeManilaV2().target
        gate_errors = basic_device_gate_errors(target=target)
        errors_on_measure = [name for name, _, _ in gate_errors if name == "measure"]
        errors_on_reset = [name for name, _, _ in gate_errors if name == "reset"]
        self.assertEqual(len(errors_on_measure), 0)
        self.assertEqual(len(errors_on_reset), 0)
        self.assertEqual(len(gate_errors), 23)
