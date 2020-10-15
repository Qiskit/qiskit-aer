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
UnitarySimulator Gate Fusion Tests
"""
from qiskit import assemble, transpile
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info import Operator
from qiskit.providers.aer import UnitarySimulator


class UnitaryFusionTests:
    """UnitarySimulator fusion tests."""

    SIMULATOR = UnitarySimulator()

    def fusion_options(self, enabled=None, threshold=None, verbose=None):
        """Return default backend_options dict."""
        backend_options = self.BACKEND_OPTS.copy()
        if enabled is not None:
            backend_options['fusion_enable'] = enabled
        if verbose is not None:
            backend_options['fusion_verbose'] = verbose
        if threshold is not None:
            backend_options['fusion_threshold'] = threshold
        return backend_options

    def fusion_metadata(self, result):
        """Return fusion metadata dict"""
        metadata = result.results[0].metadata
        return metadata.get('fusion', {})

    def test_fusion_theshold(self):
        """Test fusion threhsold settings work."""
        seed = 12345
        threshold = 4
        backend_options = self.fusion_options(enabled=True, threshold=threshold)

        with self.subTest(msg='below fusion threshold'):
            circuit = QuantumVolume(threshold - 1, seed=seed)
            circuit = transpile(circuit, self.SIMULATOR)
            qobj = assemble([circuit], shots=1)
            result = self.SIMULATOR.run(
                qobj, **backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertSuccess(result)
            self.assertFalse(meta.get('applied'))

        with self.subTest(msg='at fusion threshold'):
            circuit = QuantumVolume(threshold, seed=seed)
            circuit = transpile(circuit, self.SIMULATOR)
            qobj = assemble([circuit], shots=1)
            result = self.SIMULATOR.run(
                qobj, **backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertSuccess(result)
            self.assertFalse(meta.get('applied'))

        with self.subTest(msg='above fusion threshold'):
            circuit = QuantumVolume(threshold + 1, seed=seed)
            circuit = transpile(circuit, self.SIMULATOR)
            qobj = assemble([circuit], shots=1)
            result = self.SIMULATOR.run(
                qobj, **backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertSuccess(result)
            self.assertTrue(meta.get('applied'))

    def test_fusion_disable(self):
        """Test Fusion enable/disable option"""
        seed = 2233
        circuit = QuantumVolume(4, seed=seed)
        circuit = transpile(circuit, self.SIMULATOR)
        qobj = assemble([circuit], shots=1)

        with self.subTest(msg='test fusion enable'):
            backend_options = self.fusion_options(enabled=True, threshold=1)
            result = self.SIMULATOR.run(
                qobj, **backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertSuccess(result)
            self.assertTrue(meta.get('applied'))

        with self.subTest(msg='test fusion disable'):
            backend_options = self.fusion_options(enabled=False, threshold=1)
            result = self.SIMULATOR.run(
                qobj, **backend_options).result()
            meta = self.fusion_metadata(result)

            self.assertSuccess(result)
            self.assertFalse(meta.get('applied'))

    def test_fusion_output(self):
        """Test Fusion returns same final unitary"""
        seed = 54321
        circuit = QuantumVolume(4, seed=seed)
        circuit = transpile(circuit, self.SIMULATOR)
        qobj = assemble([circuit], shots=1)

        options_disabled = self.fusion_options(enabled=False, threshold=1)
        result_disabled = self.SIMULATOR.run(
            qobj, **options_disabled).result()
        self.assertSuccess(result_disabled)

        options_enabled = self.fusion_options(enabled=True, threshold=1)
        result_enabled = self.SIMULATOR.run(
            qobj, **options_enabled).result()
        self.assertSuccess(result_enabled)

        unitary_no_fusion = Operator(result_disabled.get_unitary(0))
        unitary_fusion = Operator(result_enabled.get_unitary(0))
        self.assertEqual(unitary_no_fusion, unitary_fusion)
