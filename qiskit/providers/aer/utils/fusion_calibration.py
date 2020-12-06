from qiskit import QuantumCircuit, assemble
from qiskit.quantum_info import random_unitary
from qiskit.providers.aer import QasmSimulator

def calibrate_fusion(**kwargs):
    
    simulator = None
    method = 'statevector'

    if 'base_qubit' in kwargs:
        base_qubit = kwargs['base_qubit']
        del kwargs['base_qubit']
    else:
        base_qubit = 20

    if 'max_qubit' in kwargs:
        max_qubit = kwargs['max_qubit']
        del kwargs['max_qubit']
    else:
        max_qubit = 5

    if 'base_qubit' in kwargs:
        base_qubit = kwargs['base_qubit']
        del kwargs['base_qubit']
    else:
        base_qubit = 20

    if 'repetitions' in kwargs:
        repetitions = kwargs['repetitions']
        del kwargs['repetitions']
    else:
        repetitions = 10
    
    if 'simulator' in kwargs:
        simulator = kwargs['simulator']
        del kwargs['simulator']
    else:
        simulator = QasmSimulator()

    if 'fusion_enable' in kwargs:
        del kwargs['fusion_enable']

    config = {}
    
    # check diag costs
    for qubit in range(1, max_qubit * 2 + 1):
      circ = QuantumCircuit(base_qubit)
      for i in range(0, repetitions):
        qubits = [ q % base_qubit for q in range(i, i + qubit) ]
        circ.diagonal([ 1, -1 ] * (2 ** (qubit - 1)), qubits) 
      qobj = assemble(circ)
      result = QasmSimulator().run(qobj, fusion_enable=False, **kwargs).result()
      time_taken = float(result.to_dict()['time_taken'])
      config['fusion_cost.diag.{0}'.format(qubit)] = time_taken
    
    for qubit in range(1, max_qubit + 1):
      circ = QuantumCircuit(base_qubit)
      for i in range(0, repetitions):
        qubits = [ q % base_qubit for q in range(i, i + qubit) ]
        circ.unitary(random_unitary(2 ** qubit), qubits)
      qobj = assemble(circ)
      result = QasmSimulator().run(qobj, fusion_enable=False, **kwargs).result()
      time_taken = float(result.to_dict()['time_taken'])
      config['fusion_cost.mat.{0}'.format(qubit)] = time_taken
    
    return config
