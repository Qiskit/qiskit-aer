import numpy as np
import math
from qiskit.circuit.library import FourierChecking, GraphState, HiddenLinearFunction, IQP, QuantumVolume, QFT #PhaseEstimation

from benchmark2.simulator_benchmark import SimulatorBenchmarkSuite

DEFAULT_APPS = [
    'fourier_checking',
    'graph_state',
    'hidden_linear_function',
    'iqp',
    'quantum_volume',
    'phase_estimation'
    ]

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

def new_diagonal_init(self, diag):
        """Check types"""
        # Check if diag has type "list"
        if not isinstance(diag, list):
            raise QiskitError("The diagonal entries are not provided in a list.")
        # Check if the right number of diagonal entries is provided and if the diagonal entries
        # have absolute value one.
        num_action_qubits = math.log2(len(diag))
        if num_action_qubits < 1 or not num_action_qubits.is_integer():
            raise QiskitError("The number of diagonal entries is not a positive power of 2.")
        for z in diag:
            if np.isscalar(z):
                continue
            try:
                complex(z)
            except TypeError:
                raise QiskitError("Not all of the diagonal entries can be converted to "
                                  "complex numbers.")
            if not np.abs(z) - 1 < _EPS:
                raise QiskitError("A diagonal entry has not absolute value one.")
        # Create new gate.
        super().__init__("diagonal", int(num_action_qubits), diag)    

from qiskit.circuit import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit, QuantumRegister
from qiskit.exceptions import QiskitError

_EPS = 1e-10  # global variable used to chop very small numbers to zero
class NewDiagonalGate(Gate):
    def __init__(self, diag):
        if not isinstance(diag, list):
            raise QiskitError("The diagonal entries are not provided in a list.")
        num_action_qubits = math.log2(len(diag))
        if num_action_qubits < 1 or not num_action_qubits.is_integer():
            raise QiskitError("The number of diagonal entries is not a positive power of 2.")
        for z in diag:
            if np.isscalar(z):
                continue
            try:
                complex(z)
            except TypeError:
                raise QiskitError("Not all of the diagonal entries can be converted to "
                                  "complex numbers.")
            if not np.abs(z) - 1 < _EPS:
                raise QiskitError("A diagonal entry has not absolute value one.")
        super().__init__("diagonal", int(num_action_qubits), [])
        self._params.append(diag)

    def _define(self):
        diag_circuit = self._dec_diag()
        gate = diag_circuit.to_instruction()
        q = QuantumRegister(self.num_qubits)
        diag_circuit = QuantumCircuit(q)
        diag_circuit.append(gate, q[:])
        self.definition = diag_circuit

    def validate_parameter(self, parameter):
        if isinstance(parameter, complex):
            return complex(parameter)
        else:
            return complex(super().validate_parameter(parameter))

    def inverse(self):
        return DiagonalGate([np.conj(entry) for entry in self.params])

    def _dec_diag(self):
        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q)
        diag_phases = [cmath.phase(z) for z in self.params]
        n = len(self.params)
        while n >= 2:
            angles_rz = []
            for i in range(0, n, 2):
                diag_phases[i // 2], rz_angle = _extract_rz(diag_phases[i], diag_phases[i + 1])
                angles_rz.append(rz_angle)
            num_act_qubits = int(np.log2(n))
            contr_qubits = q[self.num_qubits - num_act_qubits + 1:self.num_qubits]
            target_qubit = q[self.num_qubits - num_act_qubits]
            circuit.ucrz(angles_rz, contr_qubits, target_qubit)
            n //= 2
        return circuit

def _extract_rz(phi1, phi2):
    phase = (phi1 + phi2) / 2.0
    z_angle = phi2 - phi1
    return phase, z_angle

def new_diagonal(self, diag, qubit):
    if isinstance(qubit, QuantumRegister):
        qubit = qubit[:]
    if not isinstance(qubit, list):
        raise QiskitError("The qubits must be provided as a list "
                          "(also if there is only one qubit).")
    if not isinstance(diag, list):
        raise QiskitError("The diagonal entries are not provided in a list.")
    num_action_qubits = math.log2(len(diag))
    if not len(qubit) == num_action_qubits:
        raise QiskitError("The number of diagonal entries does not correspond to"
                          " the number of qubits.")
    return self.append(NewDiagonalGate(diag), qubit)

from qiskit.extensions.quantum_initializer import DiagonalGate
QuantumCircuit.diagonal = new_diagonal

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

    def fourier_checking(self, qubit, repetition):
        f = [-1, 1] * (2 ** (qubit - 1))
        g = [1, -1] * (2 ** (qubit - 1))
        return FourierChecking(f, g)
    
    def graph_state(self, qubit, repetition):
        a = np.reshape([0] * (qubit ** 2), [qubit] * 2)
        for _ in range(qubit):
            while True:
                i = np.random.randint(0, qubit)
                j = np.random.randint(0, qubit)
                if a[i][j] == 0:
                    a[i][j] = 1
                    a[j][i] = 1
                    break
        return  GraphState(a)
    
    def hidden_linear_function(self, qubit, repetition):
        a = np.reshape([0] * (qubit ** 2), [qubit] * 2)
        for _ in range(qubit):
            while True:
                i = np.random.randint(0, qubit)
                j = np.random.randint(0, qubit)
                if a[i][j] == 0:
                    a[i][j] = 1
                    a[j][i] = 1
                    break
        return HiddenLinearFunction(a)

    def iqp(self, qubit, repetition):
        interactions = np.random.randint(-1024, 1024, (qubit, qubit))
        for i in range(qubit):
            for j in range(i + 1, qubit):
                interactions[j][i] = interactions[i][j]
        return IQP(interactions).decompose()

    def quantum_volume(self, qubit, repetition):
        return QuantumVolume(qubit).decompose()

    def phase_estimation(self, qubit, repetition):
        if qubit < 6:
            raise ValueError('qubit is too small: {0}'.format(qubit))
        return PhaseEstimation(2, QuantumVolume(qubit - 2).decompose()).decompose()
    
if __name__ == "__main__":
    ParticularQuantumCircuits().run_manual()