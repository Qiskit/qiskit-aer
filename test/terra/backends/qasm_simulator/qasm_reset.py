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

from test.terra.reference import ref_reset
from qiskit.compiler import assemble
from qiskit.providers.aer import QasmSimulator


class QasmResetTests:
    """QasmSimulator reset tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test reset
    # ---------------------------------------------------------------------
    def test_reset_deterministic(self):
        """Test QasmSimulator reset with for circuits with deterministic counts"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 100
        circuits = ref_reset.reset_circuits_deterministic(final_measure=True)
        targets = ref_reset.reset_counts_deterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)

    def test_reset_nondeterministic(self):
        """Test QasmSimulator reset with for circuits with non-deterministic counts"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 4000
        circuits = ref_reset.reset_circuits_nondeterministic(
            final_measure=True)
        targets = ref_reset.reset_counts_nondeterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_reset_sampling_opt(self):
        """Test sampling optimization"""
        shots = 4000
        circuits = ref_reset.reset_circuits_sampling_optimization()
        targets = ref_reset.reset_counts_sampling_optimization(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_repeated_resets(self):
        """Test repeated reset operations"""
        shots = 100
        circuits = ref_reset.reset_circuits_repeated()
        targets = ref_reset.reset_counts_repeated(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.assertTrue(getattr(result, 'success', False))
        self.compare_counts(result, circuits, targets, delta=0)
