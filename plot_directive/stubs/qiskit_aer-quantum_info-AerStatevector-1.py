from numpy import sqrt
from qiskit.quantum_info import Statevector
sv=Statevector([1/sqrt(2), 0, 0, -1/sqrt(2)])
sv.draw(output='hinton')