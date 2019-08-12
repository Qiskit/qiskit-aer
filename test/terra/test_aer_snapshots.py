import unittest

from qiskit import *

from qiskit.providers.aer import *
from qiskit.providers.aer.extensions import *

class TestSnapshot(unittest.TestCase):

    def setUp(self):
        #Test circuit set up
        self.q = QuantumRegister(2)
        self.c = ClassicalRegister(2)
        self.qc = QuantumCircuit(self.q,self.c)
        self.qc.h(0)
        self.qc.cx(0,1)

    def test_snapshot_statevector(self):
        #Adding snapshot
        self.qc.snapshot_statevector('statevector_snapshot')

        #Execute circuit on statevector_simulator
        sv_backend = Aer.get_backend('statevector_simulator')
        job = execute(self.qc, sv_backend)
        result = job.result().get_statevector(self.qc)
        data = job.result().data(0)

        #Checking snapshot_statevector is created
        self.assertIn('snapshots', data)
        self.assertIn('statevector', data['snapshots'])
        self.assertIn('statevector_snapshot', data['snapshots']['statevector'])

    def test_snapshot_statevector_qasm(self):
        #Adding snapshot
        self.qc.measure(self.q, self.c)
        self.qc.snapshot_statevector('statevector_snapshot')

        #Execute circuit on QASM backend
        qasm_backend = Aer.get_backend('qasm_simulator')
        BACKEND_OPTS = {'method' : 'statevector'}
        job = execute(self.qc, qasm_backend, backend_options=BACKEND_OPTS)
        data = job.result().data(0)

        #Checking snapshot_statevector is created
        self.assertIn('snapshots', data)
        self.assertIn('statevector', data['snapshots'])
        self.assertIn('statevector_snapshot', data['snapshots']['statevector'])


    def test_snapshot_stabilizer(self):
        #Adding measurement and snapshot
        self.qc.measure(self.q, self.c)
        self.qc.snapshot_stabilizer('stabilizer_snapshot')

        #Execute on qasm_simulator with stabilizer method
        qasm_backend = Aer.get_backend('qasm_simulator')
        BACKEND_OPTS = {'method' : 'stabilizer'}
        job = execute(self.qc, qasm_backend, backend_options=BACKEND_OPTS, shots=10)
        data = job.result().data(0)

        #Checking snapshot_stabilizer is created
        self.assertIn('snapshots', data)
        self.assertIn('stabilizer', data['snapshots'])
        self.assertIn('stabilizer_snapshot', data['snapshots']['stabilizer'])


    def test_snapshot_density_matrix(self):
        #Test circuit
        self.qc.measure(self.q,self.c)
        self.qc.snapshot_density_matrix('density_matrix_snapshot')

        #Execute on qasm_simulator with density_matrix method
        qasm_backend = Aer.get_backend('qasm_simulator')
        BACKEND_OPTS = {'method' : 'density_matrix'}
        job = execute(self.qc, qasm_backend, shots=10)
        data = job.result().data(0)

        #Checking snapshot_density_matrix is created
        self.assertIn('snapshots', data)
        self.assertIn('density_matrix', data['snapshots'])
        self.assertIn('density_matrix_snapshot', data['snapshots']['stabilizer'])


    def test_snapshot_probabilities(self):
        #Adding measurement and snapshot
        self.qc.measure(self.q, self.c)
        self.qc.snapshot_probabilities('probabilities_snapshot', qubits=[self.q[0]])

        #Execute on qasm_simulator
        qasm_backend = Aer.get_backend('qasm_simulator')
        job = execute(self.qc, qasm_backend, shots=10)
        data = job.result().data(0)

        #Checking snapshot_probabilities is created
        self.assertIn('snapshots', data)
        self.assertIn('probabilities', data['snapshots'])
        self.assertIn('probabilities_snapshot', data['snapshots']['probabilities'])

        def test_snapshot_expectation_value(self):
            #Adding measurement and snapshot
            self.qc.measure(self.q, self.c)
            self.qc.snapshot_expectation_value('expectation_value_snapshot', qubits=[self.q[0]])

            #Execute on qasm_simulator
            qasm_backend = Aer.get_backend('qasm_simulator')
            job = execute(self.qc, qasm_backend, shots=10)
            data = job.result().data(0)

            #Checking snapshot_expectation_value is created
            self.assertIn('snapshots', data)
            self.assertIn('expectation_value', data['snapshots'])
            self.assertIn('expectation_value_snapshot', data['snapshots']['probabilities'])
