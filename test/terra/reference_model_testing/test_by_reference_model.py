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

class TestByReferenceModel(common.QiskitAerTestCase):

    def setUp(self):
        self.qasm_sim = QasmSimulator()
        self.den_sim = DensityMatrixSimulator()


    def test_probs_snapshot(self, qc=None):

        if qc == None:
            qc = common.generate_random_circuit(2+np.random.randint(4), 1+np.random.randint(15),
                                                self.den_sim.get_supported_gates())

        # The following lines implement a hack,
        # which allows to test the probabilities snapshot
        # for circuits that contain measurements.
        # The probabilities snapshot provides a vector of probabilities
        # for each possible state of the classical registers.
        # We would like to obtain the probabilities regardless
        # of the classical registers.
        # To this end, we add a dummy qubit to the circuit,
        # and measure it to every classical register,
        # resulting in a single state of classical registers.
        q_dummy = QuantumRegister(1, 'q_dummy')
        qc.add(q_dummy)
        for classical_register in qc.get_cregs()['cr']:
            qc.measure(q_dummy[0], classical_register)
        
        self.log.debug(qc.qasm())

        qobj = compile(qc, self.qasm_sim, shots=10000, seed=1)
        den_result = self.den_sim.run(qobj)

        # Add a probabilities snapshot at the end of the circuit
        qobj.experiments[0].instructions.append(qobj_utils.qobj_snapshot_item(snapshot_type='probabilities', label='final', qubits=list(range(len(qc.get_qregs()['qr'])))))
        qasm_result = self.qasm_sim.run(qobj).result()
        
        den_probs = den_result.extract_probs()
        self.log.debug(den_probs)
        qasm_probs = qasm_result.get_snapshots(qc)['probabilities']['final'][0]['value']
        self.log.debug(qasm_probs)
        
        self.assertDictAlmostEqual(den_probs, qasm_probs, delta=1e-2)
        

    def test_state_snapshot(self, qc=None):

        if qc == None:
            qc = common.generate_random_circuit(2+np.random.randint(4), 1+np.random.randint(15),
                                                self.den_sim.get_supported_gates())
        self.log.debug(qc.qasm())

        qobj = compile(qc, self.qasm_sim, shots=10000, seed=1)
        den_result = self.den_sim.run(qobj)

        qobj.experiments[0].instructions.append(qobj_utils.qobj_snapshot_item(snapshot_type='statevector', label='final'))
        qasm_result = self.qasm_sim.run(qobj).result()

        nqubits = len(qc.get_qregs()['qr'])

        # all statevectors, from all shots
        states = qasm_result.get_snapshots(qc)['statevector']['final']
        
        qasm_den_mat = DensityMatrix([QuantumState(state) for state in states],
                                     ProbabilityDistribution([1]*len(states), not_normalized=True))
        self.assertTrue(is_close(den_result.rho, qasm_den_mat.rho, rel_tol=1e-2, abs_tol=1e-2))


if __name__ == '__main__':
    unittest.main()
