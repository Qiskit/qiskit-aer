from test.terra.utils import common
from qiskit_aer.utils import qobj_utils
import unittest
import numpy as np
import math
import random

from density_matrix_simulator import DensityMatrixSimulator
from qstructs import DensityMatrix, QuantumState, ProbabilityDistribution
from qstructs import is_close, get_extended_ops, randcomplex
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import compile
from qiskit_aer.backends import QasmSimulator
from qiskit.qobj import QobjItem

class TestMatrixSnapshot(common.QiskitAerTestCase):

    def setUp(self):
        self.qasm_sim = QasmSimulator()
        self.den_sim = DensityMatrixSimulator()

    def test_matrix_snapshot(self, qc=None, params=None):

        if qc == None:
            qc = common.generate_random_circuit(2+np.random.randint(4), 1+np.random.randint(15),
                                                self.den_sim.get_supported_gates())

        # The following lines implement a hack,
        # which allows to test the matrix snapshot
        # for circuits that contain measurements.
        # The matrix snapshot provides expectation value
        # for each possible state of the classical registers.
        # We would like to obtain the expectation value regardless
        # of the classical registers.
        # To this end, we add a dummy qubit to the circuit,
        # and measure it to every classical register,
        # resulting in a single state of classical registers.
        q_dummy = QuantumRegister(1, 'q_dummy')
        qc.add(q_dummy)
        for classical_register in qc.get_cregs()['cr']:
            qc.measure(q_dummy[0], classical_register)
        
        self.log.debug(qc.qasm())
        nqubits = len(qc.get_qregs()['qr'])
 
        qobj = compile(qc, self.qasm_sim, shots=50000, seed=1)

        if params == None:
            num_of_components = 1 + np.random.randint(2)
            params = []
             
            for _ in range(num_of_components):
                num_of_mats = 1 + np.random.randint(2)
                coeff = 10 * randcomplex(1)[0]
                blocks = []
                
                for _ in range(num_of_mats):
                    num_of_involved_qubits = 1 + np.random.randint(nqubits-1)
                    mat_size = 2**(num_of_involved_qubits)
                    mat = np.zeros([mat_size, mat_size], dtype=complex)
                    
                    for row in range(mat_size):
                        mat[row, :] = 10 * randcomplex(mat_size)
                        qubits = random.sample(range(nqubits), num_of_involved_qubits)
                        blocks.append([qubits, mat])    
                        
                params.append([coeff, blocks])

        self.log.debug(params)

        den_result = self.den_sim.run(qobj)
        den_expectation_value = den_result.observable(params)

        # Add a matrix snapshot at the end of the circuit
        qobj.experiments[0].instructions.append(
            qobj_utils.qobj_snapshot_item(snapshot_type='expval_matrix', label='final', params=params))

        qasm_result = self.qasm_sim.run(qobj).result()
        qasm_expectation_value = qasm_result.get_snapshots(qc)['expectation_value']['final'][0]['value']

        self.assertTrue(is_close(qasm_expectation_value, den_expectation_value, rel_tol=0.3, abs_tol=1e-2))


if __name__ == '__main__':
    unittest.main()
