import numpy as np
import math
from qiskit.circuit.library import FourierChecking, GraphState, HiddenLinearFunction, IQP, QuantumVolume, QFT #PhaseEstimation

from benchmark2.simulator_benchmark import SimulatorBenchmarkSuite

DEFAULT_APPS = {
    'fourier_checking': 10,
    'graph_state': 10,
    'hidden_linear_function': 10,
    'iqp': 10,
    'quantum_volume': 1,
    'phase_estimation': 1
    }

DEFAULT_QUBITS = SimulatorBenchmarkSuite.DEFAULT_QUBITS

DEFAULT_RUNTIME = [
    SimulatorBenchmarkSuite.RUNTIME_STATEVECTOR_CPU,
    SimulatorBenchmarkSuite.RUNTIME_MPS_CPU,
    SimulatorBenchmarkSuite.RUNTIME_STATEVECTOR_GPU
    ]

DEFAULT_MEASUREMENT_METHODS = [
    SimulatorBenchmarkSuite.MEASUREMENT_SAMPLING
    ]

DEFAULT_MEASUREMENT_COUNTS = SimulatorBenchmarkSuite.DEFAULT_MEASUREMENT_COUNTS

DEFAULT_NOISE_MODELS = [
    SimulatorBenchmarkSuite.NOISE_IDEAL
]

from qiskit import QuantumCircuit, QuantumRegister
from typing import Optional

class PhaseEstimation(QuantumCircuit):
    def __init__(self,
                 num_evaluation_qubits: int,
                 unitary: QuantumCircuit,
                 iqft: Optional[QuantumCircuit] = None,
                 name: str = 'QPE') -> None:
        qr_eval = QuantumRegister(num_evaluation_qubits, 'eval')
        qr_state = QuantumRegister(unitary.num_qubits, 'q')
        super().__init__(qr_eval, qr_state, name=name)

        if iqft is None:
            iqft = QFT(num_evaluation_qubits, inverse=True, do_swaps=False)

        self.h(qr_eval)  # hadamards on evaluation qubits

        ctr_unitary = unitary.control()
        for j in range(num_evaluation_qubits):  # controlled powers
            u = QuantumCircuit(qr_eval, qr_state)
            u.append(ctr_unitary, [j] + qr_state[:])
            u = u.decompose()
            for _ in range(2**j):
                self._data += u._data
        self.append(iqft.decompose(), qr_eval[:])  # final QFT

class ParticularQuantumCircuits(SimulatorBenchmarkSuite):

    def __init__(self,
                 apps = DEFAULT_APPS,
                 qubits = DEFAULT_QUBITS,
                 runtime_names = DEFAULT_RUNTIME,
                 measures = DEFAULT_MEASUREMENT_METHODS, 
                 measure_counts = DEFAULT_MEASUREMENT_COUNTS, 
                 noise_model_names=DEFAULT_NOISE_MODELS):
        super().__init__('particular_quantum_circuits', 
                         apps, qubits=qubits, 
                         runtime_names=runtime_names, 
                         measures=measures, 
                         measure_counts=measure_counts, 
                         noise_model_names=noise_model_names)
    
    def repeat(self, circ, repeats):
        if repeats is not None and repeats > 1:
            circ = circ.repeat(repeats).decompose()
        return circ
    
    def fourier_checking(self, qubit, repeats):
        if qubit > 20:
            raise ValueError('qubit is too small: {0}'.format(qubit))
        f = [-1, 1] * (2 ** (qubit - 1))
        g = [1, -1] * (2 ** (qubit - 1))
        return self.repeat(FourierChecking(f, g), repeats)
    
    def graph_state(self, qubit, repeats):
        a = np.reshape([0] * (qubit ** 2), [qubit] * 2)
        for _ in range(qubit):
            while True:
                i = np.random.randint(0, qubit)
                j = np.random.randint(0, qubit)
                if a[i][j] == 0:
                    a[i][j] = 1
                    a[j][i] = 1
                    break
        return self.repeat(GraphState(a), repeats)
    
    def hidden_linear_function(self, qubit, repeats):
        a = np.reshape([0] * (qubit ** 2), [qubit] * 2)
        for _ in range(qubit):
            while True:
                i = np.random.randint(0, qubit)
                j = np.random.randint(0, qubit)
                if a[i][j] == 0:
                    a[i][j] = 1
                    a[j][i] = 1
                    break
        return self.repeat(HiddenLinearFunction(a), repeats)

    def iqp(self, qubit, repeats):
        interactions = np.random.randint(-1024, 1024, (qubit, qubit))
        for i in range(qubit):
            for j in range(i + 1, qubit):
                interactions[j][i] = interactions[i][j]
        return self.repeat(IQP(interactions).decompose(), repeats)

    def quantum_volume(self, qubit, repeats):
        return self.repeat(QuantumVolume(qubit).decompose(), repeats)

    def phase_estimation(self, qubit, repeats):
        if qubit < 6:
            raise ValueError('qubit is too small: {0}'.format(qubit))
        return self.repeat(PhaseEstimation(2, QuantumVolume(qubit - 2).decompose()).decompose(), repeats)
    
if __name__ == "__main__":
    ParticularQuantumCircuits().run_manual()