import test.terra.common as common
import unittest
import numpy as np

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
        
        qc = common.generate_random_circuit(2+np.random.randint(8), 1+np.random.randint(12), list(self.den_sim.gate2mats.keys()))
        print(qc.qasm())

        qobj = compile(qc, self.qasm_sim)
        den_result = self.den_sim.run(qobj)

        qobj.experiments[0].instructions.append(QobjItem(name='snapshot', type='probabilities', label='final', params=list(range(len(qc.get_qregs()['q0'])))))
        qasm_result = self.qasm_sim.run(qobj).result()
        
        self.verify_probs(den_result, qasm_result, qc)


if __name__ == '__main__':
    unittest.main()
