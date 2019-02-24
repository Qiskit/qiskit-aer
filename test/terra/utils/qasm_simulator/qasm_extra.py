# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QasmSimulator Integration Tests
"""

from test.terra.utils import common
from test.terra.utils import ref_unitary_gate
from qiskit.providers.aer import QasmSimulator


class QasmExtraTests(common.QiskitAerTestCase):
    """QasmSimulator additional tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test unitary gate qobj instruction
    # ---------------------------------------------------------------------
    def test_unitary_gate_real(self):
        """Test unitary qobj instruction with real matrices."""
        shots = 100
        qobj = ref_unitary_gate.unitary_gate_circuits_real_deterministic(final_measure=True)
        qobj.config.shots = shots
        circuits = [experiment.header.name for experiment in qobj.experiments]
        targets = ref_unitary_gate.unitary_gate_counts_real_deterministic(shots)
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_unitary_gate_complex(self):
        """Test unitary qobj instruction with complex matrices."""
        shots = 100
        qobj = ref_unitary_gate.unitary_gate_circuits_complex_deterministic(final_measure=True)
        qobj.config.shots = shots
        circuits = [experiment.header.name for experiment in qobj.experiments]
        targets = ref_unitary_gate.unitary_gate_counts_complex_deterministic(shots)
        result = self.SIMULATOR.run(qobj).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)
