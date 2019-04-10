import numpy as np
import pprint
import qiskit
from qiskit import Aer, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, QiskitError
from qiskit.providers.aer import StatevectorSimulator
from qiskit.qobj import QasmQobjInstruction

def initialize_qobj_direct():

    print("Initializing Qobj by adding QasmQobjInstruction:")
    # Construct an quantum circuit - |+++>
    qr = QuantumRegister(3)
    cr = ClassicalRegister(3)
    circ = QuantumCircuit(qr, cr)
    circ.h(qr[0])
    circ.h(qr[1])
    circ.h(qr[2])

    print ("start with |+++> state: ")

    print (circ)

    # Select the QasmSimulator from the Aer provider
    simulator = Aer.get_backend('statevector_simulator')

    # Execute and get counts
    #result = execute(circ, simulator).result()
    qobj = qiskit.compile(circ, backend=simulator)
    print (qobj)

    result = simulator.run(qobj).result()
    statevector = result.get_statevector(circ)
    counts = result.get_counts(circ)

    print ("statevector:", statevector)
    print ("counts:", counts)

    #Execute after reset&initialize - 1 qubit
    for qubit in range(3):
        print ("--------------------------------------------------------")
        print ("--------------------------------------------------------")
        print ("reset: qubit", qubit)
        qobj = qiskit.compile(circ, backend=simulator)
        qobj.experiments[0].instructions.append(QasmQobjInstruction(name='reset', qubits=[qubit]))
        print (qobj)

        result = simulator.run(qobj).result()
        statevector = result.get_statevector(circ)
        counts = result.get_counts(circ)

        print ("statevector:", statevector)
        print ("counts:", counts)

        print("--------------------------------------------------------")
        print ("initialize: qubit", qubit)
        qobj = qiskit.compile(circ, backend=simulator)
        qobj.experiments[0].instructions.append(QasmQobjInstruction(name='initialize', qubits=[qubit], params=[[0,0],[1,0]]))
        print (qobj)

        result = simulator.run(qobj).result()
        statevector = result.get_statevector(circ)
        counts = result.get_counts(circ)

        print ("statevector:", statevector)
        print ("counts:", counts)


    #Execute after reset&initialize - 2 qubits
    for qubit_i in range(3):
        for qubit_j in range(3):
            if (qubit_i != qubit_j):
                print ("--------------------------------------------------------")
                print ("--------------------------------------------------------")
                print ("reset: qubits i, j: ", qubit_i, qubit_j)
                qobj = qiskit.compile(circ, backend=simulator)
                qobj.experiments[0].instructions.append(QasmQobjInstruction(name='reset', qubits=[qubit_i, qubit_j]))
                print (qobj)
                result = simulator.run(qobj).result()
                statevector = result.get_statevector(circ)
                counts = result.get_counts(circ)

                print ("statevector:", statevector)
                print ("counts:", counts)

                print("--------------------------------------------------------")
                print("plain initialize: qubits i, j: ", qubit_i, qubit_j)
                qobj = qiskit.compile(circ, backend=simulator)
                qobj.experiments[0].instructions.append(QasmQobjInstruction(name='initialize', qubits=[qubit_i], params=[[0, 0],[1, 0]]))
                qobj.experiments[0].instructions.append(QasmQobjInstruction(name='initialize', qubits=[qubit_j], params=[[1, 0],[0, 0]]))
                print(qobj)

                result = simulator.run(qobj).result()
                statevector = result.get_statevector(circ)
                counts = result.get_counts(circ)

                print("statevector:", statevector)
                print("counts:", counts)

                plain_statevector = statevector

                print ("--------------------------------------------------------")
                print ("initialize: qubits i, j: ", qubit_i, qubit_j)
                qobj = qiskit.compile(circ, backend=simulator)
                qobj.experiments[0].instructions.append(QasmQobjInstruction(name='initialize', qubits=[qubit_i, qubit_j],\
                                                                            params=[[0, 0], [1, 0], [0, 0], [0, 0]]))
                print (qobj)

                result = simulator.run(qobj).result()
                statevector = result.get_statevector(circ)
                counts = result.get_counts(circ)

                print ("statevector:", statevector)
                print ("counts:", counts)

                print(statevector == plain_statevector)

    #Execute after reset&initialize - 3 qubits
    for qubit_i in range(3):
        for qubit_j in range(3):
            for qubit_k in range(3):
                if ((qubit_i != qubit_j) & (qubit_i != qubit_k) & (qubit_k != qubit_j)):
                    print ("--------------------------------------------------------")
                    print ("--------------------------------------------------------")
                    print ("reset: qubits i, j, k: ", qubit_i, qubit_j, qubit_k)
                    qobj = qiskit.compile(circ, backend=simulator)
                    qobj.experiments[0].instructions.append(QasmQobjInstruction(name='reset', qubits=[qubit_i, qubit_j, qubit_k]))
                    print (qobj)
                    result = simulator.run(qobj).result()
                    statevector = result.get_statevector(circ)
                    counts = result.get_counts(circ)

                    print ("statevector:", statevector)
                    print ("counts:", counts)

                    print ("--------------------------------------------------------")
                    print ("plain initialize: qubits i, j, k: ", qubit_i, qubit_j, qubit_k)
                    qobj = qiskit.compile(circ, backend=simulator)
                    qobj.experiments[0].instructions.append(QasmQobjInstruction(name='initialize', qubits=[qubit_i], params=[[0, 0], [1, 0]]))
                    qobj.experiments[0].instructions.append(QasmQobjInstruction(name='initialize', qubits=[qubit_j], params=[[1, 0], [0, 0]]))
                    qobj.experiments[0].instructions.append(QasmQobjInstruction(name='initialize', qubits=[qubit_k], params=[[0.70710678, 0], [-0.70710678, 0]]))
                    print (qobj)

                    result = simulator.run(qobj).result()
                    statevector = result.get_statevector(circ)
                    counts = result.get_counts(circ)

                    plain_statevector = statevector
                    print ("statevector:", statevector)
                    print ("counts:", counts)

                    print("--------------------------------------------------------")
                    print("initialize: qubits i, j, k: ", qubit_i, qubit_j, qubit_k)
                    qobj = qiskit.compile(circ, backend=simulator)
                    qobj.experiments[0].instructions.append(
                        QasmQobjInstruction(name='initialize', qubits=[qubit_i,qubit_j,qubit_k],
                                            params=[[0, 0], [0.70710678, 0], [0, 0], [0, 0],
                                                    [0, 0], [-0.70710678, 0], [0, 0], [0, 0]]))
                    print (qobj)

                    result = simulator.run(qobj).result()
                    statevector = result.get_statevector(circ)
                    counts = result.get_counts(circ)

                    print("statevector:", statevector)
                    print("counts:", counts)

                    print (statevector == plain_statevector)

    print ("-------------------------------------------")
    print ("-------------------------------------------")
    print ("-------------------------------------------")
    print ("start with Bell state: ")

    # Construct a quantum circuit - Bell
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2)
    circ = QuantumCircuit(qr, cr)
    circ.h(qr[0])
    circ.cx(qr[0], qr[1])

    print (circ)

    # Select the QasmSimulator from the Aer provider
    simulator = Aer.get_backend('statevector_simulator')

    # Execute and get counts
    #result = execute(circ, simulator).result()
    qobj = qiskit.compile(circ, backend=simulator)
    print (qobj)

    result = simulator.run(qobj).result()
    statevector = result.get_statevector(circ)
    counts = result.get_counts(circ)

    print ("statevector:", statevector)
    print ("counts:", counts)

    print("--------------------------------------------------------")
    print("--------------------------------------------------------")
    print("reset: qubit", 0)
    qobj = qiskit.compile(circ, backend=simulator)
    qobj.experiments[0].instructions.append(QasmQobjInstruction(name='reset', qubits=[0]))
    # print (qobj)

    result = simulator.run(qobj).result()
    statevector = result.get_statevector(circ)
    counts = result.get_counts(circ)

    print("statevector:", statevector)
    print("counts:", counts)

    print("--------------------------------------------------------")
    print("--------------------------------------------------------")

    for times in range(20):
        print("initialize: qubit", 0)
        qobj = qiskit.compile(circ, backend=simulator)
        qobj.experiments[0].instructions.append(QasmQobjInstruction(name='initialize', qubits=[0], params=[[0.70710678, 0], [0.70710678, 0]]))
        # print (qobj)

        result = simulator.run(qobj).result()
        statevector = result.get_statevector(circ)
        counts = result.get_counts(circ)

        print("statevector:", statevector)
        #print("counts:", counts)

    print("--------------------------------------------------------")
    print("--------------------------------------------------------")

    for times in range(20):
        print("initialize: qubit", 0)
        qobj = qiskit.compile(circ, backend=simulator)
        qobj.experiments[0].instructions.append(QasmQobjInstruction(name='initialize', qubits=[0], params=[[0.70710678, 0], [-0.70710678, 0]]))
        # print (qobj)

        result = simulator.run(qobj).result()
        statevector = result.get_statevector(circ)
        counts = result.get_counts(circ)

        print("statevector:", statevector)
        #print("counts:", counts)
