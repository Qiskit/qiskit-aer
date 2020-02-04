"""
NoiseModel class integration tests
"""

import unittest
from test.terra import common
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.noiseerror import NoiseError
from qiskit.providers.aer.noise.errors import depolarizing_error
from qiskit.providers.aer.utils import remap_noise_model


class TestNoiseRemapper(common.QiskitAerTestCase):
    """Testing remap_noise_model function"""

    def test_raises_duplicate_qubits(self):
        """Test duplicate qubits raises exception"""
        model = NoiseModel()
        self.assertRaises(NoiseError, remap_noise_model, model, [[0, 1], [2, 1]], warnings=False)
        model = NoiseModel()
        error = depolarizing_error(0.5, 1)
        model.add_quantum_error(error, ['u3'], [2], False)
        self.assertRaises(NoiseError, remap_noise_model, model, [[3, 2]], warnings=False)

    def test_remap_all_qubit_quantum_errors(self):
        """Test remapper doesn't effect all-qubit quantum errors."""
        model = NoiseModel()
        error1 = depolarizing_error(0.5, 1)
        error2 = depolarizing_error(0.5, 2)
        model.add_all_qubit_quantum_error(error1, ['u3'], False)
        model.add_all_qubit_quantum_error(error2, ['cx'], False)

        remapped_model = remap_noise_model(model, [[0, 1], [1, 0]], warnings=False)
        self.assertEqual(model, remapped_model)

    def test_remap_quantum_errors(self):
        """Test remapping of quantum errors."""
        model = NoiseModel()
        error1 = depolarizing_error(0.5, 1)
        error2 = depolarizing_error(0.5, 2)
        model.add_quantum_error(error1, ['u3'], [0], False)
        model.add_quantum_error(error2, ['cx'], [1, 2], False)

        remapped_model = remap_noise_model(model, [[0, 1], [1, 2], [2, 0]], warnings=False)
        target = NoiseModel()
        target.add_quantum_error(error1, ['u3'], [1], False)
        target.add_quantum_error(error2, ['cx'], [2, 0], False)
        self.assertEqual(remapped_model, target)

    def test_remap_nonlocal_quantum_errors(self):
        """Test remapping of non-local quantum errors."""
        model = NoiseModel()
        error1 = depolarizing_error(0.5, 1)
        error2 = depolarizing_error(0.5, 2)
        model.add_nonlocal_quantum_error(error1, ['u3'], [0], [1], False)
        model.add_nonlocal_quantum_error(error2, ['cx'], [1, 2], [3, 0], False)

        remapped_model = remap_noise_model(model, [[0, 1], [1, 2], [2, 0]], warnings=False)
        target = NoiseModel()
        target.add_nonlocal_quantum_error(error1, ['u3'], [1], [2], False)
        target.add_nonlocal_quantum_error(error2, ['cx'], [2, 0], [3, 1], False)
        self.assertEqual(remapped_model, target)

    def test_remap_all_qubit_readout_errors(self):
        """Test remapping of all-qubit readout errors."""
        model = NoiseModel()
        error1 = [[0.9, 0.1], [0.5, 0.5]]
        model.add_all_qubit_readout_error(error1, False)

        remapped_model = remap_noise_model(model, [[0, 1], [1, 2], [2, 0]], warnings=False)
        self.assertEqual(remapped_model, model)

    def test_remap_readout_errors(self):
        """Test remapping of readout errors."""
        model = NoiseModel()
        error1 = [[0.9, 0.1], [0.5, 0.5]]
        error2 = [[0.8, 0.2, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0.1, 0.9]]
        model.add_readout_error(error1, [1], False)
        model.add_readout_error(error2, [0, 2], False)

        remapped_model = remap_noise_model(model, [[0, 1], [1, 2], [2, 0]], warnings=False)
        target = NoiseModel()
        target.add_readout_error(error1, [2], False)
        target.add_readout_error(error2, [1, 0], False)
        self.assertEqual(remapped_model, target)

    def test_reduce_noise_model(self):
        """Test reduction mapping of noise model."""
        error1 = depolarizing_error(0.5, 1)
        error2 = depolarizing_error(0.5, 2)
        roerror1 = [[0.9, 0.1], [0.5, 0.5]]
        roerror2 = [[0.8, 0.2, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0.1, 0.9]]

        model = NoiseModel()
        model.add_all_qubit_quantum_error(error1, ['u3'], False)
        model.add_quantum_error(error1, ['u3'], [1], False)
        model.add_nonlocal_quantum_error(error2, ['cx'], [2, 0], [3, 1], False)
        model.add_all_qubit_readout_error(roerror1, False)
        model.add_readout_error(roerror2, [0, 2], False)

        remapped_model = remap_noise_model(model, [0, 1, 2], discard_qubits=True, warnings=False)
        target = NoiseModel()
        target.add_all_qubit_quantum_error(error1, ['u3'], False)
        target.add_quantum_error(error1, ['u3'], [1], False)
        target.add_all_qubit_readout_error(roerror1, False)
        target.add_readout_error(roerror2, [0, 2], False)
        self.assertEqual(remapped_model, target)

    def test_reduce_remapped_noise_model(self):
        """Test reduction and remapping of noise model."""
        error1 = depolarizing_error(0.5, 1)
        error2 = depolarizing_error(0.5, 2)
        roerror1 = [[0.9, 0.1], [0.5, 0.5]]
        roerror2 = [[0.8, 0.2, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0.1, 0.9]]

        model = NoiseModel()
        model.add_all_qubit_quantum_error(error1, ['u3'], False)
        model.add_quantum_error(error1, ['u3'], [1], False)
        model.add_nonlocal_quantum_error(error2, ['cx'], [2, 0], [3, 1], False)
        model.add_all_qubit_readout_error(roerror1, False)
        model.add_readout_error(roerror2, [0, 2], False)

        remapped_model = remap_noise_model(model, [2, 0, 1], discard_qubits=True, warnings=False)
        target = NoiseModel()
        target.add_all_qubit_quantum_error(error1, ['u3'], False)
        target.add_quantum_error(error1, ['u3'], [2], False)
        target.add_all_qubit_readout_error(roerror1, False)
        target.add_readout_error(roerror2, [1, 0], False)
        self.assertEqual(remapped_model, target)


if __name__ == '__main__':
    unittest.main()
