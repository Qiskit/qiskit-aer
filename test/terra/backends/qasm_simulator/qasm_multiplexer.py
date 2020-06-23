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

from test.terra.reference import ref_multiplexer
from qiskit import execute
from qiskit.providers.aer import QasmSimulator


class QasmMultiplexerTests:
    """QasmSimulator multiplexer gate tests in default basis."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test multiplexer-cx-gate
    # ---------------------------------------------------------------------
    def test_multiplexer_cx_gate_deterministic(self):
        """Test multiplxer cx-gate circuits compiling to backend default basis_gates."""
        shots = 100
        circuits = ref_multiplexer.multiplexer_cx_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_multiplexer.multiplexer_cx_gate_counts_deterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots, backend_options=self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_multiplexer_cx_gate_nondeterministic(self):
        """Test multiplexer cx-gate circuits compiling to backend default basis_gates."""
        shots = 4000
        circuits = ref_multiplexer.multiplexer_cx_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_multiplexer.multiplexer_cx_gate_counts_nondeterministic(shots)
        job = execute(circuits, self.SIMULATOR, shots=shots, backend_options=self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test multiplexer-gate
    # ---------------------------------------------------------------------
    def test_multiplexer_cxx_gate_deterministic(self):
        """Test multiplexer-gate gate circuits """
        shots = 100
        circuits = ref_multiplexer.multiplexer_ccx_gate_circuits_deterministic(
            final_measure=True)
        targets = ref_multiplexer.multiplexer_ccx_gate_counts_deterministic(
            shots)
        job = execute(circuits, self.SIMULATOR, shots=shots,
                      backend_options=self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_multiplexer_cxx_gate_nondeterministic(self):
        """Test multiplexer ccx-gate gate circuits """
        shots = 4000
        circuits = ref_multiplexer.multiplexer_ccx_gate_circuits_nondeterministic(
            final_measure=True)
        targets = ref_multiplexer.multiplexer_ccx_gate_counts_nondeterministic(
            shots)
        job = execute(circuits, self.SIMULATOR, shots=shots,
                      backend_options=self.BACKEND_OPTS)
        result = job.result()
        self.assertSuccess(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
