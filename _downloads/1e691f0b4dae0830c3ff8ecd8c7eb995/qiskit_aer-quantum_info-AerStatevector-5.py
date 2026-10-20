from qiskit.quantum_info import Statevector

psi = Statevector.from_label('-0')
print(psi.to_dict())