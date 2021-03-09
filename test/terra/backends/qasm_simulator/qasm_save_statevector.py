# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
QasmSimulator Integration Tests for SaveStatevector instruction
"""

import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, assemble
from qiskit.providers.aer import QasmSimulator


class QasmSaveStatevectorTests:
    """QasmSimulator SaveStatevector instruction tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    def test_save_statevector(self):
        """Test save statevector for instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'matrix_product_state', 'extended_stabilizer'
        ]

        # Stabilizer test circuit
        circ = QuantumCircuit(3)
        circ.h(0)
        circ.sdg(0)
        circ.cx(0, 1)
        circ.cx(0, 2)

        # Target statevector
        target = qi.Statevector(circ)

        # Add save to circuit
        label = 'sv'
        circ.save_statevector(label)

        # Run
        opts = self.BACKEND_OPTS.copy()
        qobj = assemble(circ, self.SIMULATOR)
        result = self.SIMULATOR.run(qobj, **opts).result()
        method = opts.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            data = result.data(0)
            self.assertIn(label, data)
            value = qi.Statevector(result.data(0)[label])
            self.assertAlmostEqual(value, target)

    def test_save_statevector_conditional(self):
        """Test conditional save statevector instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'matrix_product_state', 'extended_stabilizer'
        ]

        # Stabilizer test circuit
        label = 'sv'
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.sdg(0)
        circ.cx(0, 1)
        circ.measure_all()
        circ.save_statevector(label, conditional=True)

        # Target statevector
        target = {'0x0': qi.Statevector([1, 0, 0, 0]),
                  '0x3': qi.Statevector([0, 0, 0, -1j])}

        # Run
        opts = self.BACKEND_OPTS.copy()
        qobj = assemble(circ, self.SIMULATOR, shots=10)
        result = self.SIMULATOR.run(qobj, **opts).result()
        method = opts.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            data = result.data(0)
            self.assertIn(label, data)
            for key, vec in data[label].items():
                self.assertIn(key, target)
                self.assertAlmostEqual(qi.Statevector(vec), target[key])

    def test_save_statevector_pershot(self):
        """Test pershot save statevector instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'matrix_product_state', 'extended_stabilizer'
        ]

        # Stabilizer test circuit
        circ = QuantumCircuit(1)
        circ.x(0)
        circ.reset(0)
        circ.h(0)
        circ.sdg(0)

        # Target statevector
        target = qi.Statevector(circ)

        # Add save
        label = 'sv'
        circ.save_statevector(label, pershot=True)

        # Run
        shots = 10
        opts = self.BACKEND_OPTS.copy()
        qobj = assemble(circ, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, **opts).result()
        method = opts.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            data = result.data(0)
            self.assertIn(label, data)
            value = result.data(0)[label]
            self.assertEqual(len(value), shots)
            for vec in value:
                self.assertAlmostEqual(qi.Statevector(vec), target)

    def test_save_statevector_pershot_conditional(self):
        """Test pershot conditional save statevector instruction"""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'matrix_product_state', 'extended_stabilizer'
        ]

        # Stabilizer test circuit
        circ = QuantumCircuit(1)
        circ.x(0)
        circ.reset(0)
        circ.h(0)
        circ.sdg(0)

        # Target statevector
        target = qi.Statevector(circ)

        # Add save
        label = 'sv'
        circ.save_statevector(label, pershot=True, conditional=True)
        circ.measure_all()

        # Run
        shots = 10
        opts = self.BACKEND_OPTS.copy()
        qobj = assemble(circ, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, **opts).result()
        method = opts.get('method', 'automatic')
        if method not in SUPPORTED_METHODS:
            self.assertFalse(result.success)
        else:
            self.assertTrue(result.success)
            data = result.data(0)
            self.assertIn(label, data)
            value = result.data(0)[label]
            self.assertIn('0x0', value)
            self.assertEqual(len(value['0x0']), shots)
            for vec in value['0x0']:
                self.assertAlmostEqual(qi.Statevector(vec), target)
