# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Noise pass classes tests
"""

from qiskit.providers.aer.noise.passes import RelaxationNoisePass
from test.terra.common import QiskitAerTestCase

from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.test.mock import FakeLagos
from qiskit.transpiler import InstructionDurations


class TestRelaxationNoisePass(QiskitAerTestCase):
    """Testing RelaxationNoisePass class"""

    def test_default(self):
        """Test noises on all ops."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        backend = FakeLagos()

        sched_circ = transpile(qc, backend, scheduling_method='alap')

        delay_pass = RelaxationNoisePass(
            t1s=[backend.properties().t1(q) for q in range(backend.configuration().num_qubits)],
            t2s=[backend.properties().t2(q) for q in range(backend.configuration().num_qubits)],
            instruction_durations=InstructionDurations.from_backend(backend)
        )
        noisy_circ = delay_pass(sched_circ)
        self.assertEqual(9, noisy_circ.count_ops()["quantum_channel"])

    def test_ops_option(self):
        """Test noises only on delays."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        backend = FakeLagos()

        sched_circ = transpile(qc, backend, scheduling_method='alap')

        delay_pass = RelaxationNoisePass(
            t1s=[backend.properties().t1(q) for q in range(backend.configuration().num_qubits)],
            t2s=[backend.properties().t2(q) for q in range(backend.configuration().num_qubits)],
            instruction_durations=InstructionDurations.from_backend(backend),
            ops="delay",
        )
        noisy_circ = delay_pass(sched_circ)
        self.assertEqual(6, noisy_circ.count_ops()["quantum_channel"])
