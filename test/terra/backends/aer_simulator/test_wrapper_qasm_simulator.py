# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
StatevectorSimulator Integration Tests
"""

from ddt import ddt, data

from qiskit_aer import QasmSimulator, AerError
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods


@ddt
class TestQasmSimulator(SimulatorTestCase):
    """QasmSimulator backend wrapper tests."""

    BACKEND = QasmSimulator

    @supported_methods(["statevector", "density_matrix"])
    def test_legacy_methods(self, method, device):
        """Test legacy device method options."""
        backend = self.backend()
        # GPU_cuStateVec is converted to GPU
        if device == "GPU_cuStateVec":
            device = "GPU"
        # GPU_batch is converted to GPU
        if device == "GPU_batch":
            device = "GPU"
        legacy_method = f"{method}_{device.lower()}"
        backend.set_options(method=legacy_method)
        self.assertEqual(backend.options.method, method)
        self.assertEqual(backend.options.device, device)

    @data("unitary", "superop")
    def test_unsupported_methods(self, method):
        """Test unsupported AerSimulator method raises AerError."""
        backend = self.backend()
        with self.assertRaises(AerError):
            backend.set_options(method=method)
