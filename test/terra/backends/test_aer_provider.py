# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Tests for AerProvider and AerSimulator
"""

from qiskit import *
from qiskit.circuit.library import *
from qiskit_aer import *

from test.terra.common import QiskitAerTestCase


class TestAerProvider(QiskitAerTestCase):
    """Tests for AerProvider."""

    def test_enum_backends(self):
        """Test backends enumrated by AerProvider."""
        provider = AerProvider()
        for backend in provider.backends():
            if "aer_simulator" in backend.name:
                method = getattr(backend.options, "method", "automatic")
                device = getattr(backend.options, "device", "CPU")
                if method == "automatic":
                    self.assertEqual(backend.name, "aer_simulator")
                else:
                    name = "aer_simulator"
                    name += f"_{method}"
                    if device != "CPU":
                        name += f"_{device}".lower()
                    self.assertEqual(backend.name, name)

    def test_get_backend(self):
        """Test running backend from AerProvider."""
        provider = AerProvider()
        backend = provider.get_backend("aer_simulator_density_matrix")
        circuit = transpile(QuantumVolume(4, 10, seed=0), backend=backend, optimization_level=0)
        circuit.measure_all()
        results = backend.run(circuit, shots=100, seed_simulator=0).result()

        self.assertEqual(results.results[0].metadata["method"], "density_matrix")
        self.assertEqual(results.results[0].metadata["device"], "CPU")

    def test_backend_name_by_set_option(self):
        """Test backend name of AerSimulator by set_option."""
        backend = AerSimulator()
        if "GPU" in backend.available_devices():
            backend.set_option("device", "GPU")
            backend.set_option("method", "density_matrix")
            self.assertEqual(backend.name, "aer_simulator_density_matrix_gpu")
        else:
            backend.set_option("device", "Thrust")
            backend.set_option("method", "density_matrix")
            self.assertEqual(backend.name, "aer_simulator_density_matrix_thrust")
