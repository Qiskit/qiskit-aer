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
Matrix product state integration tests
This set of tests is copied from qasm_measure.py, but without some
functionality that has not been supported yet (specifically the 'iden'
operation).
"""

from test.terra.reference import ref_measure
from qiskit.compiler import assemble
from qiskit.providers.aer import QasmSimulator

class QasmMatrixProductStateMeasureTests:
    """QasmSimulator matrix_product_state measure tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test measure
    # ---------------------------------------------------------------------
    def test_measure_deterministic_with_sampling(self):
        """Test QasmSimulator measure with deterministic counts with sampling"""
        shots = 100
        circuits = ref_measure.measure_circuits_deterministic(
            allow_sampling=True)
        targets = ref_measure.measure_counts_deterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)
        # Test sampling was enabled
        for res in result.results:
            self.assertIn("measure_sampling", res.metadata)
            self.assertEqual(res.metadata["measure_sampling"], True)
            
    def test_measure_nondeterministic_with_sampling(self):
        """Test QasmSimulator measure with non-deterministic counts with sampling"""
        shots = 2000
        circuits = ref_measure.measure_circuits_nondeterministic(
            allow_sampling=True)
        targets = ref_measure.measure_counts_nondeterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
        # Test sampling was enabled
        for res in result.results:
            self.assertIn("measure_sampling", res.metadata)
            self.assertEqual(res.metadata["measure_sampling"], True)

    # ---------------------------------------------------------------------
    # Test multi-qubit measure qobj instruction
    # ---------------------------------------------------------------------
    def test_measure_nondeterministic_multi_qubit_with_sampling(self):
        """Test QasmSimulator measure with non-deterministic counts"""
        shots = 2000
        circuits = ref_measure.multiqubit_measure_circuits_nondeterministic(
            allow_sampling=True)
        targets = ref_measure.multiqubit_measure_counts_nondeterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
        # Test sampling was enabled
        for res in result.results:
            self.assertIn("measure_sampling", res.metadata)
            self.assertEqual(res.metadata["measure_sampling"], True)

