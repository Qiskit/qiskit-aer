# This code is part of Qiskit.
#
# (C) Copyright IBM Corp. 2017 and later.
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

from test.terra.reference import ref_measure
from qiskit.compiler import assemble
from qiskit.providers.aer import QasmSimulator


class QasmMeasureTests:
    """QasmSimulator measure tests."""

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

    def test_measure_deterministic_without_sampling(self):
        """Test QasmSimulator measure with deterministic counts without sampling"""
        shots = 100
        circuits = ref_measure.measure_circuits_deterministic(
            allow_sampling=False)
        targets = ref_measure.measure_counts_deterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

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

    def test_measure_nondeterministic_without_sampling(self):
        """Test QasmSimulator measure with nin-deterministic counts without sampling"""
        shots = 2000
        circuits = ref_measure.measure_circuits_nondeterministic(
            allow_sampling=False)
        targets = ref_measure.measure_counts_nondeterministic(shots)
        qobj = assemble(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    # ---------------------------------------------------------------------
    # Test multi-qubit measure qobj instruction
    # ---------------------------------------------------------------------
    def test_measure_deterministic_multi_qubit_with_sampling(self):
        """Test QasmSimulator multi-qubit measure with deterministic counts with sampling"""
        shots = 100
        qobj = ref_measure.measure_circuits_qobj_deterministic(
            allow_sampling=True)
        qobj.config.shots = shots
        circuits = [experiment.header.name for experiment in qobj.experiments]
        targets = ref_measure.measure_counts_qobj_deterministic(shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_measure_deterministic_multi_qubit_without_sampling(self):
        """Test QasmSimulator multi-qubit measure with deterministic counts without sampling"""
        shots = 100
        qobj = ref_measure.measure_circuits_qobj_deterministic(
            allow_sampling=False)
        qobj.config.shots = shots
        circuits = [experiment.header.name for experiment in qobj.experiments]
        targets = ref_measure.measure_counts_qobj_deterministic(shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_measure_nondeterministic_multi_qubit_with_sampling(self):
        """Test QasmSimulator measure with non-deterministic counts"""
        shots = 2000
        qobj = ref_measure.measure_circuits_qobj_nondeterministic(
            allow_sampling=True)
        qobj.config.shots = shots
        circuits = [experiment.header.name for experiment in qobj.experiments]
        targets = ref_measure.measure_counts_qobj_nondeterministic(shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)

    def test_measure_nondeterministic_multi_qubit_without_sampling(self):
        """Test QasmSimulator measure with non-deterministic counts"""
        shots = 2000
        qobj = ref_measure.measure_circuits_qobj_nondeterministic(
            allow_sampling=False)
        qobj.config.shots = shots
        circuits = [experiment.header.name for experiment in qobj.experiments]
        targets = ref_measure.measure_counts_qobj_nondeterministic(shots)
        result = self.SIMULATOR.run(
            qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
