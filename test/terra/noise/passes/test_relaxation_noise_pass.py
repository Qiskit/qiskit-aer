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
RelaxationNoisePass class tests
"""

from qiskit.providers.aer.noise.passes import RelaxationNoisePass

from qiskit.circuit import QuantumCircuit, Delay
from qiskit.compiler import transpile
from qiskit.test.mock import FakeLagos
from qiskit.transpiler import TranspilerError
from test.terra.common import QiskitAerTestCase


class TestRelaxationNoisePass(QiskitAerTestCase):
    """Testing RelaxationNoisePass class"""

    def test_default_with_scheduled_circuit(self):
        """Test adding noises to all ops in a scheduled circuit."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        backend = FakeLagos()

        sched_circ = transpile(qc, backend, scheduling_method='alap')

        noise_pass = RelaxationNoisePass(
            t1s=[backend.properties().t1(q) for q in range(backend.configuration().num_qubits)],
            t2s=[backend.properties().t2(q) for q in range(backend.configuration().num_qubits)],
            dt=backend.configuration().dt
        )
        noisy_circ = noise_pass(sched_circ)
        self.assertEqual(9, noisy_circ.count_ops()["quantum_channel"])

    def test_default_with_non_scheduled_circuit(self):
        """Test never adding noises to non-scheduled circuit."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        noise_pass = RelaxationNoisePass(
            t1s=[0.10, 0.11],
            t2s=[0.20, 0.21],
            dt=0.01,
        )
        with self.assertWarns(UserWarning):
            noisy_circ = noise_pass(qc)

        self.assertEqual(qc, noisy_circ)

    def test_raise_if_supplied_invalid_ops(self):
        with self.assertRaises(TranspilerError):
            RelaxationNoisePass(
                t1s=[1],
                t2s=[1],
                dt=1,
                op_types="delay",  # str is invalid
            )

    def test_ops_option_with_scheduled_circuit(self):
        """Test adding noises only to delays in a scheduled circuit."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        backend = FakeLagos()

        sched_circ = transpile(qc, backend, scheduling_method='alap')

        delay_pass = RelaxationNoisePass(
            t1s=[backend.properties().t1(q) for q in range(backend.configuration().num_qubits)],
            t2s=[backend.properties().t2(q) for q in range(backend.configuration().num_qubits)],
            dt=backend.configuration().dt,
            op_types=Delay,
        )
        noisy_circ = delay_pass(sched_circ)
        self.assertEqual(6, noisy_circ.count_ops()["quantum_channel"])
