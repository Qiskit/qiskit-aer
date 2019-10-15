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
QasmSimulator Integration Tests
"""

from test.terra.reference import ref_initialize
from qiskit.compiler import assemble
from qiskit.providers.aer import QasmSimulator


class QasmInitializeTests:
    """QasmSimulator initialize tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test initialize
    # ---------------------------------------------------------------------
    def test_initialize_1(self):
        """Test QasmSimulator initialize"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 2000
        circuits = ref_initialize.initialize_circuits_1(final_measure=True)
        targets = ref_initialize.initialize_counts_1(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_initialize_2(self):
        """Test QasmSimulator initializes"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 2000
        circuits = ref_initialize.initialize_circuits_2(final_measure=True)
        targets = ref_initialize.initialize_counts_2(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_initialize_sampling_opt(self):
        """Test sampling optimization"""
        shots = 2000
        circuits = ref_initialize.initialize_sampling_optimization()
        targets = ref_initialize.initialize_counts_sampling_optimization(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
