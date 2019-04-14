from qiskit import *
from qiskit import Aer, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, QiskitError
from qiskit.providers.aer import StatevectorSimulator
from qiskit.compiler import assemble_circuits,  RunConfig
from qiskit.qobj import QasmQobjInstruction
import numpy as np

def initialize_QuantumCircuit():
    print("--------------------------------------")
    print("--------------------------------------")
    print("--------------------------------------")
    print("Initializing QuantumCircuit:")
    qr = QuantumRegister(2)

    # Select the QasmSimulator from the Aer provider
    simulator = Aer.get_backend('statevector_simulator')

    # Method 1
    circ1 = QuantumCircuit(qr)
    circ1.initialize([0, 0, 0, 1], qr[:])
    print(circ1)

    qobj = assemble_circuits(circ1, run_config=RunConfig(shots=1000))
    sim_job = simulator.run(qobj)
    # Getting the result
    result = sim_job.result()

    print(qobj)
    statevector = result.get_statevector(circ1)
    print ("statevector of circ1:", statevector)

    # Method 2
    circ2 = QuantumCircuit(qr)
    circ2.initialize([0, 0, 0, 1], [qr[0], qr[1]])
    print(circ2)

    qobj = assemble_circuits(circ2, run_config=RunConfig(shots=1000))
    sim_job = simulator.run(qobj)
    # Getting the result
    result = sim_job.result()

    statevector = result.get_statevector(circ2)
    print ("statevector of circ2:", statevector)

    # Method 3
    circ3 = QuantumCircuit(qr)
    circ3.initialize([0, 1], [qr[0]])
    circ3.initialize([0, 1], [qr[1]])
    print(circ3)

    qobj = assemble_circuits(circ3, run_config=RunConfig(shots=1000))
    sim_job = simulator.run(qobj)
    # Getting the result
    result = sim_job.result()

    print(qobj)
    statevector = result.get_statevector(circ3)
    print ("statevector of circ3:", statevector)

    # Implementation
    circ0 = QuantumCircuit(qr)
    circ0.reset(qr[0])
    circ0.x(qr[0])
    circ0.reset(qr[1])
    circ0.x(qr[1])
    print(circ0)

    # Execute and get counts
    qobj = assemble_circuits(circ0, run_config=RunConfig(shots=1000))
    sim_job = simulator.run(qobj)
    # Getting the result
    result = sim_job.result()
    statevector = result.get_statevector(circ0)
    print ("statevector of circ0:", statevector)

    # assemble into a qobj
    qobj = assemble_circuits([circ0, circ1, circ2, circ3])
    print ("*******************************")
    print (qobj)
    print ("*******************************")

    qr = QuantumRegister(3)

    circ4 = QuantumCircuit(qr)
    circ4.h(qr[0])
    circ4.h(qr[1])
    circ4.h(qr[2])
    print(circ4)

    qobj = assemble_circuits(circ4, run_config=RunConfig(shots=1000))
    sim_job = simulator.run(qobj)
    # Getting the result
    result = sim_job.result()

    statevector = result.get_statevector(circ4)
    print ("statevector of circ4:", statevector)

    qobj = assemble_circuits(circ4, run_config=RunConfig(shots=1000))
    print(qobj)
    #qobj.experiments[0].instructions.append(QasmQobjInstruction(name='initialize', qubits=[0], params=[[0,0],[1,0]]))
    #print(qobj)

    #result = sim_job.result()
    #statevector = result.get_statevector(circ4)
    #print ("statevector of updated circ4 (after initialize):", statevector)

    circ5 = QuantumCircuit(qr)
    circ5.h(qr[0])
    circ5.h(qr[1])
    circ5.h(qr[2])
    circ5.reset (qr[0])
    print(circ5)

    qobj = assemble_circuits(circ5, run_config=RunConfig(shots=1000))
    print(qobj)
    sim_job = simulator.run(qobj)
    result = sim_job.result()
    statevector = result.get_statevector(circ5)
    print ("statevector of circ5:", statevector)


    circ6 = QuantumCircuit(qr)
    circ6.h(qr[0])
    circ6.h(qr[1])
    circ6.h(qr[2])
    circ6.initialize(np.array([0.+0.j, 1.+0.j]), [qr[0]])
    print(circ6)

    qobj = assemble_circuits(circ6, run_config=RunConfig(shots=1000))
    print(qobj)
    sim_job = simulator.run(qobj)
    result = sim_job.result()
    print(qobj)
    statevector = result.get_statevector(circ6)
    print ("statevector of circ6:", statevector)


    circ7 = QuantumCircuit(qr)
    circ7.h(qr[0])
    circ7.h(qr[1])
    circ7.h(qr[2])
    circ7.initialize(np.array([0,1], dtype=complex), [qr[0]])
    print(circ7)

    qobj = assemble_circuits(circ7, run_config=RunConfig(shots=1000))
    print(qobj)
    sim_job = simulator.run(qobj)
    result = sim_job.result()
    print(qobj)
    statevector = result.get_statevector(circ7)
    print ("statevector of circ7:", statevector)

# initialize_QuantumCircuit()