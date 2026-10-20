from qiskit.quantum_info import Statevector

psi = Statevector.from_label('+0')

# Probabilities for measuring both qubits
probs = psi.probabilities([0, 1])
print('probs: {}'.format(probs))

# Probabilities for measuring both qubits
# but swapping qubits 0 and 1 in output
probs_swapped = psi.probabilities([1, 0])
print('Swapped probs: {}'.format(probs_swapped))