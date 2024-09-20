# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Tests for Aer BackendConfiguration
"""

from test.terra.common import QiskitAerTestCase

from ddt import data, ddt
from qiskit_aer.backends import AerSimulator, QasmSimulator, StatevectorSimulator, UnitarySimulator


@ddt
class TestBackendConfiguration(QiskitAerTestCase):
    """Tests for Aer BackendConfiguration."""

    @data(AerSimulator(), StatevectorSimulator(), QasmSimulator(), UnitarySimulator())
    def test_open_pulse_in_backend_configuration(self, simulator: str):
        """Test for backend_configuration"""
        assert simulator.configuration().open_pulse is False
        assert simulator.configuration().to_dict()["open_pulse"] is False
