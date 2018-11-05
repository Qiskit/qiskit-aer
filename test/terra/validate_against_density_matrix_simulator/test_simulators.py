import test.terra.common as common
import unittest
import numpy as np
import math

from density_matrix_simulator import DensityMatrixSimulator
from qstructs import DensityMatrix, QuantumState, ProbabilityDistribution
from qstructs import is_close, state_reverse
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
        
        self.assertDictAlmostEqual(den_probs, qasm_probs, delta=1e-2)


    def test_probs_snapshot(self, qc=None):

        if qc == None:
            qc = common.generate_random_circuit(2+np.random.randint(4), 1+np.random.randint(15),
                                                self.den_sim.get_supported_gates())

        # The following lines implement a hack,
        # which allows to test the probabilities snapshot
        # for circuits that contain mwasurements.
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
        
        print(qc.qasm())

        qobj = compile(qc, self.qasm_sim, shots=10000, seed=1)
        den_result = self.den_sim.run(qobj)

        # Add a probabilities snapshot at the end of the circuit
        qobj.experiments[0].instructions.append(QobjItem(name='snapshot', type='probabilities', label='final',
                                                         params=list(range(len(qc.get_qregs()['qr'])))))
        qasm_result = self.qasm_sim.run(qobj).result()
        
        self.verify_probs(den_result, qasm_result, qc)


    def verify_states(self, den_result, qasm_result, qc):
        
        nqubits = len(qc.get_qregs()['qr'])

        # all statevectors, from all shots
        states = qasm_result.get_snapshots(qc)['state']['final']

        # adjusted_states will reflect the fact that the statevector
        # amplitudes are ordered lexicographically, where qubit 0
        # is LSB, whereas the rows and columns of the density matrix
        # are ordered with qubit 0 as MSB
        adjusted_states = []

        # Note that len(states) is not always equal to the number of shots,
        # bacause the simulator has an optimization when all measurements are
        # at the end. If the optimization occurs the 'states' consists of
        # a single state.
        for shot in range(len(states)):
            state = states[shot]
            adjusted_state = np.zeros(2**nqubits, dtype=complex)
            for basis_state in range(2**nqubits):
                adjusted_state[state_reverse(basis_state, nqubits)] = state[basis_state]
            adjusted_states.append(QuantumState(adjusted_state))
        
        qasm_den_mat = DensityMatrix(adjusted_states,
                                     ProbabilityDistribution([1]*len(adjusted_states), not_normalized=True))
        self.assertTrue(is_close(den_result.rho, qasm_den_mat.rho, rel_tol=1e-2, abs_tol=1e-2))
        

    def test_state_snapshot(self, qc=None):

        if qc == None:
            qc = common.generate_random_circuit(2+np.random.randint(4), 1+np.random.randint(15),
                                                self.den_sim.get_supported_gates())
        print(qc.qasm())

        shots = 10000
        qobj = compile(qc, self.qasm_sim, shots=shots, seed=1)
        den_result = self.den_sim.run(qobj)

        qobj.experiments[0].instructions.append(QobjItem(name='snapshot', type='state', label='final'))
        qasm_result = self.qasm_sim.run(qobj).result()

        self.verify_states(den_result, qasm_result, qc)


if __name__ == '__main__':
    unittest.main()
