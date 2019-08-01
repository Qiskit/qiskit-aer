import unittest

from qiskit import *
from qiskit.providers.aer import *

class TestSnapshot(unittest.TestCase):

    def test_snapshot_statevector(self):
        #Test circuit
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q,c)
        qc.h(0)
        qc.cx(0,1)
        qc.snapshot_statevector('statevector_snapshot')

        #Execute circuit on statevector_simulator
        sv_backend = Aer.get_backend('statevector_simulator')
        sv_job = execute(qc, sv_backend)
        sv_result = sv_job.result().get_statevector(qc)
        data = sv_job.result().data(0)

        #Checking snapshot_statevector is created
        self.assertIn('snapshots', data)
        self.assertIn('statevector', data['snapshots'])
        self.assertIn('statevector_snapshot', data['snapshots']['statevector'])


    def test_snapshot_stabilizer(self):
        #Test circuit
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q,c)
        qc.h(0)
        qc.cx(0,1)
        qc.measure(q,c)
        qc.snapshot_stabilizer('stabilizer_snapshot')

        #Execute on qasm_simulator with stabilizer method
        qasm_backend = Aer.get_backend('qasm_simulator')
        BACKEND_OPTS = {'method' : 'stabilizer'}
        job = execute(qc, qasm_backend, backend_options=BACKEND_OPTS, shots=10)
        data = job.result().data(0)

        #Checking snapshot_stabilizer is created
        self.assertIn('snapshots', data)
        self.assertIn('stabilizer', data['snapshots'])
        self.assertIn('stabilizer_snapshot', data['snapshots']['stabilizer'])


    def test_snapshot_density_matrix(self):
        #Test circuit
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q,c)
        qc.h(0)
        qc.cx(0,1)
        qc.measure(q,c)
        qc.snapshot_density_matrix('density_matrix_snapshot')

        #Execute on qasm_simulator with density_matrix method
        qasm_backend = Aer.get_backend('qasm_simulator')
        BACKEND_OPTS = {'method' : 'density_matrix'}
        job = execute(qc, qasm_backend, shots=10)
        data = job.result().data(0)

        #Checking snapshot_density_matrix is created
        self.assertIn('snapshots', data)
        self.assertIn('density_matrix', data['snapshots'])
        self.assertIn('density_matrix_snapshot', data['snapshots']['stabilizer'])
