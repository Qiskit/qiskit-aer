# This code is part of Qiskit.
#
# (C) Copyright IBM 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
AerSimulator Integration Tests
"""

from test.terra.backends.simulator_test_case import SimulatorTestCase

from qiskit.providers.fake_provider import FakeNairobi, FakeNairobiV2
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile

from qiskit_aer import AerSimulator


class TestAlgorithms(SimulatorTestCase):
    """AerSimulator algorithm tests in the default basis"""
    def setUp(self):
        super().setUp()
        self.qc = QuantumCircuit(2)
        self.qc.h(0)
        self.qc.cx(0, 1)
        self.qc.measure_all()

    def test_backend_v1(self):
        """Test from_backend_v1 works."""
        backend = FakeNairobi()
        sim_backend = AerSimulator.from_backend(backend)
        tqc = transpile(self.qc, backend)
        self.assertEqual(
            backend.run(tqc, shots=1024, seed_simulator=12345678942).result().get_counts(),
            sim_backend.run(tqc, shots=1024, seed_simulator=12345678942).result().get_counts(),
        )

    def test_backend_v2(self):
        """Test from_backend_v2 works."""
        backend = FakeNairobiV2()
        sim_backend = AerSimulator.from_backend(backend)
        tqc = transpile(self.qc, backend)
        self.assertDictAlmostEqual(
            backend.run(tqc, shots=1024, seed_simulator=12345678942).result().get_counts(),
            sim_backend.run(tqc, shots=1024, seed_simulator=12345678942).result().get_counts(),
            delta=100
        )
