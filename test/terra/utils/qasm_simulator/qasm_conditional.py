# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QasmSimulator Integration Tests
"""

from test.terra.utils import common
from test.terra.utils import ref_conditionals
from qiskit import compile
from qiskit.providers.aer import QasmSimulator


class QasmConditionalTests(common.QiskitAerTestCase):
    """QasmSimulator conditional tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    # ---------------------------------------------------------------------
    # Test conditional
    # ---------------------------------------------------------------------
    def test_conditional_1bit(self):
        """Test conditional operations on 1-bit conditional register."""
        shots = 100
        circuits = ref_conditionals.conditional_circuits_1bit(final_measure=True)
        targets = ref_conditionals.conditional_counts_1bit(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_conditional_2bit(self):
        """Test conditional operations on 2-bit conditional register."""
        shots = 100
        circuits = ref_conditionals.conditional_circuits_2bit(final_measure=True)
        targets = ref_conditionals.conditional_counts_2bit(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)
