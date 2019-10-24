from random_statevector_test import random_statevector_test

max_num_qubits = 22
max_num_gates = 200
for num_qubits in range(2, max_num_qubits+1, 2):
    for num_gates in range(10, max_num_gates+1, 20):
        random_statevector_test(num_qubits, num_gates)


