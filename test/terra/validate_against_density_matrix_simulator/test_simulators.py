import test.terra.common as common
import unittest

from density_matrix_simulator import DensityMatrixSimulator
from qstructs import DensityMatrix
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import compile
from qiskit_aer.backends import QasmSimulator

class TestSimulators(common.QiskitAerTestCase):

    def setUp(self):
        self.qasm_sim = QasmSimulator()
        self.den_sim = DensityMatrixSimulator()

    def test_simulators(self):
        
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(4)
        qc = QuantumCircuit(qreg, creg)
        qc.h(qreg[0])
        qc.cx(qreg[0], qreg[1])

        qobj = compile(qc, self.qasm_sim)
        result = self.den_sim.run(qobj)
        print(result)

