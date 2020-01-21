import numpy as np
import math
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler import assemble
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info.synthesis import two_qubit_cnot_decompose


class QuantumFourierTransformFusionSuite:
    def __init__(self):
        self.timeout = 60 * 20
        self.backend = QasmSimulator()
        num_qubits = [5, 10, 15, 20, 25]
        self.circuit = {}
        for num_qubit in num_qubits:
            for use_cu1 in [True, False]:
                circuit = self.qft_circuit(num_qubit, use_cu1)
                self.circuit[(num_qubit, use_cu1)] = assemble(circuit, self.backend, shots=1)
        self.param_names = ["Quantum Fourier Transform", "Fusion Activated", "Use cu1 gate"]
        self.params = (num_qubits, [True, False], [True, False])

    @staticmethod
    def qft_circuit(num_qubit, use_cu1):
        qreg = QuantumRegister(num_qubit,"q")
        creg = ClassicalRegister(num_qubit, "c")
        circuit = QuantumCircuit(qreg, creg)

        for i in range(num_qubit):
            circuit.h(qreg[i])

        for i in range(num_qubit):
            for j in range(i):
                l = math.pi/float(2**(i-j))
                if use_cu1:
                    circuit.cu1(l, qreg[i], qreg[j])
                else:
                    circuit.u1(l/2, qreg[i])
                    circuit.cx(qreg[i], qreg[j])
                    circuit.u1(-l/2, qreg[j])
                    circuit.cx(qreg[i], qreg[j])
                    circuit.u1(l/2, qreg[j])
            circuit.h(qreg[i])

        circuit.barrier()
        for i in range(num_qubit):
            circuit.measure(qreg[i], creg[i])

        return circuit

    def time_quantum_fourier_transform(self, num_qubit, fusion_enable, use_cu1):
        """ Benchmark QFT """
        result = self.backend.run(self.circuit[(num_qubit, use_cu1)], backend_options={'fusion_enable': fusion_enable}).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)


class RandomFusionSuite:
    def __init__(self):
        self.timeout = 60 * 20
        self.backend = QasmSimulator()
        self.param_names = ["Number of Qubits", "Fusion Activated"]
        self.params = ([5, 10, 15, 20, 25], [True, False])

    @staticmethod
    def build_model_circuit_kak(width, depth, seed=None):
        """Create quantum volume model circuit on quantum register qreg of given
        depth (default depth is equal to width) and random seed.
        The model circuits consist of layers of Haar random
        elements of U(4) applied between corresponding pairs
        of qubits in a random bipartition.
        """
        qreg = QuantumRegister(width)
        depth = depth or width

        np.random.seed(seed)
        circuit = QuantumCircuit(qreg, name="Qvolume: %s by %s, seed: %s" % (width, depth, seed))

        for _ in range(depth):
            # Generate uniformly random permutation Pj of [0...n-1]
            perm = np.random.permutation(width)

            # For each pair p in Pj, generate Haar random U(4)
            # Decompose each U(4) into CNOT + SU(2)
            for k in range(width // 2):
                U = random_unitary(4, seed).data
                for gate in two_qubit_cnot_decompose(U):
                    qs = [qreg[int(perm[2 * k + i.index])] for i in gate[1]]
                    pars = gate[0].params
                    name = gate[0].name
                    if name == "cx":
                        circuit.cx(qs[0], qs[1])
                    elif name == "u1":
                        circuit.u1(pars[0], qs[0])
                    elif name == "u2":
                        circuit.u2(*pars[:2], qs[0])
                    elif name == "u3":
                        circuit.u3(*pars[:3], qs[0])
                    elif name == "id":
                        pass  # do nothing
                    else:
                        raise Exception("Unexpected gate name: %s" % name)
        return circuit

    def time_random_transform(self, num_qubits, fusion_enable):
        circ = self.build_model_circuit_kak(num_qubits, num_qubits, 1)
        qobj = assemble(circ)
        result = self.backend.run(qobj, backend_options={'fusion_enable': fusion_enable}).result()
        if result.status != 'COMPLETED':
            raise QiskitError("Simulation failed. Status: " + result.status)