from density_matrix_simulator import DensityMatrixSimulator
from qstructs import DensityMatrix
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import compile
from qiskit_addon_qv import AerQvSimulator

qv_backend = AerQvSimulator()
den_sim = DensityMatrixSimulator()

qreg = QuantumRegister(2)
creg = ClassicalRegister(4)
qc = QuantumCircuit(qreg, creg)
qc.h(qreg[0])
qc.cx(qreg[0], qreg[1])

qobj = compile(qc, qv_backend)
result = den_sim.run(qobj)
print(result)

