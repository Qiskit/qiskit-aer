import unittest

from qiskit import *
from qiskit.providers.aer import *

class TestSnapshot(unittest.TestCase):

    def test_snapshot_statevector(self):
        #Bell state test circuit
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q,c)
        qc.h(0)
        qc.cx(0,1)

        #Execute circuit on statevector_simulator without snapshot
        sv_backend = Aer.get_backend('statevector_simulator')
        sv_job1 = execute(qc, sv_backend)
        sv_result1 = sv_job1.result().get_statevector(qc)

        #Execute circuit on statevector_simulator with snapshot
        qc.snapshot_statevector('sv_snapshot')
        sv_job2 = execute(qc, sv_backend)
        sv_result2 = sv_job2.result().get_statevector(qc)
        data = sv_job2.result().data(0)

        #Check snapshot data is correct for statevector_simulator
        self.assertIn('snapshots', data)
        self.assertIn('statevector', data['snapshots'])
        self.assertIn('sv_snapshot', data['snapshots']['statevector'])
        self.assertEqual(sv_result1.all(), sv_result2.all())






    def test_snapshot_stabilizer(self):
        pass

    def test_snapshot_densitymatrix(self):
        pass

    def test_snapshot_probabilities(self):
        pass

    def test_snapshot_expectationvalues(self):
        pass
