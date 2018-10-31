import test.terra.common as common
import unittest

from density_matrix_simulator import DensityMatrixSimulator
from qstructs import DensityMatrix
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import compile
from qiskit_aer.backends import QasmSimulator
from qiskit.qobj import QobjItem

class TestSimulators(common.QiskitAerTestCase):

    def setUp(self):
        self.qasm_sim = QasmSimulator()
        self.den_sim = DensityMatrixSimulator()

    def verify_probs(self, den_result, qasm_result, qc):
        den_probs = den_result.extract_probs()
        print(den_probs)
        qasm_probs = qasm_result.get_snapshots(qc)['probabilities']['final'][0]['values']
        print(qasm_probs)
        self.assertDictAlmostEqual(den_probs, qasm_probs)

    def test_simulators(self):
        
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(4)
        qc = QuantumCircuit(qreg, creg)
        qc.h(qreg[0])
        qc.cx(qreg[0], qreg[1])

        qobj = compile(qc, self.qasm_sim)
        den_result = self.den_sim.run(qobj)
        print(den_result)

        qobj.experiments[0].instructions.append(QobjItem(name='snapshot', type='probabilities', label='final', params=[0, 1]))
        qasm_result = self.qasm_sim.run(qobj).result()
        print(qasm_result)
        
        self.verify_probs(den_result, qasm_result, qc)
        

        

