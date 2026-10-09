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
AerSimulator Integration Tests
"""

from test.terra.backends.simulator_test_case import SimulatorTestCase

from qiskit import QuantumCircuit, ClassicalRegister


class TestConditionalErrors(SimulatorTestCase):
    def test_infinite_run_error(self):
        backend = self.backend(method="statevector", device="CPU")
        backend.set_options(max_parallel_experiments=0)

        main_circ = QuantumCircuit(1)
        creg_0 = ClassicalRegister(1)
        main_circ.add_register(creg_0)
        main_circ.measure(0, creg_0[0])
        main_circ.x(0)
        with main_circ.if_test((creg_0[0], 0)) as else_1:
            pass
        with else_1:
            pass
        main_circ.measure_active()

        result = backend.run(main_circ, shots=1).result()
        self.assertSuccess(result)
