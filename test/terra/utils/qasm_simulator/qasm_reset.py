# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QasmSimulator Integration Tests
"""

from test.terra.utils import common
from test.terra.utils import ref_reset
from qiskit import compile
from qiskit.providers.aer import QasmSimulator


class QasmResetTests(common.QiskitAerTestCase):
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
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0)

    def test_reset_nondeterministic(self):
        """Test QasmSimulator reset with for circuits with non-deterministic counts"""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 2000
        circuits = ref_reset.reset_circuits_nondeterministic(final_measure=True)
        targets = ref_reset.reset_counts_nondeterministic(shots)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options=self.BACKEND_OPTS).result()
        self.is_completed(result)
        self.compare_counts(result, circuits, targets, delta=0.05 * shots)
