from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.providers.basic_provider import BasicSimulator

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.swap(1, 2)

backend = BasicSimulator()
transpiled1 = transpile(qc, backend, optimization_level=0)
transpiled2 = transpile(qc, backend, optimization_level=2)

# Get statevectors accounting for layout changes
sv1 = Statevector.from_circuit(transpiled1)
sv2 = Statevector.from_circuit(transpiled2)

# These will be equivalent up to global phase
print(sv1.equiv(sv2))  # True