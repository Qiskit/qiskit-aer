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

from test.terra.reference import ref_readout_noise

from qiskit.compiler import assemble
from qiskit.providers.aer import QasmSimulator


class QasmReadoutNoiseTests:
    """QasmSimulator readout error noise model tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}

    def test_readout_noise(self):
        """Test simulation with classical readout error noise model."""
        # For statevector output we can combine deterministic and non-deterministic
        # count output circuits
        shots = 2000
        circuits = ref_readout_noise.readout_error_circuits()
        noise_models = ref_readout_noise.readout_error_noise_models()
        targets = ref_readout_noise.readout_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models,
                                                targets):
            qobj = assemble(circuit, self.SIMULATOR, shots=shots)
            result = self.SIMULATOR.run(
                qobj,
                backend_options=self.BACKEND_OPTS,
                noise_model=noise_model).result()
            self.is_completed(result)
            self.compare_counts(result, [circuit], [target], delta=0.05*shots)
