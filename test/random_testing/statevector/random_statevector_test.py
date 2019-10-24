#import qiskit
from qiskit import Aer
from qiskit.providers.aer import aerbackend
#from providers.aer.backends import aerbackend
from qiskit.providers.aer.utils import qobj_utils
from qiskit.providers.aer.utils.qobj_utils import snapshot_instr
from qiskit.qobj.models.qasm import QasmQobjInstruction, QasmQobjExperiment
from qiskit.qobj.models.base import QobjHeader
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
import copy
from copy import deepcopy
from pprint import pprint
import random

from qiskit.assembler import assemble_circuits
from create_circuit import create_circuit

def random_statevector_test(num_qubits=4, num_gates=10, seed=100000000):
    backend_qasm = Aer.get_backend('qasm_simulator')

    if seed == 100000000:
        seed = random.choice(range(seed))
    print("running test: num_qubits=" + str(num_qubits) + ", num_gates=" + str(num_gates) + ", seed = " + str(seed))

    qasm_qobj = create_circuit(num_qubits, num_gates, seed)
    new_instr = snapshot_instr(snapshot_type= "statevector",
                               label= "sv")
    qasm_qobj.experiments[0].instructions.append(new_instr)
        
    BACKEND_OPTS_QASM = {"method": "statevector"}

    job_sim_qasm = backend_qasm.run(qasm_qobj, backend_options=BACKEND_OPTS_QASM)
    result_qasm = job_sim_qasm.result()
    #print(result_qasm)
    res_qasm = result_qasm.results
    #print(res_qasm)

    sv_qasm = res_qasm[0].data.snapshots.statevector['sv'][0]
    #print(">>> qasm statevector = " + str(sv_qasm))

#--------------

    BACKEND_OPTS_TN = {"method": "matrix_product_state"}

    job_sim_TN = backend_qasm.run(qasm_qobj, backend_options=BACKEND_OPTS_TN)
    result_TN = job_sim_TN.result()
    res_TN = result_TN.results

    sv_mps = res_TN[0].data.snapshots.statevector['sv'][0]
    #print(">>> MPS statevector = " + str(sv_mps))
    
    failure = 0
    threshold = 1e-5
# compare results of the two statevectors
    for key in range(len(sv_mps)):
        diff0 = sv_mps[key][0] - sv_qasm[key][0]
        diff1 = sv_mps[key][1] - sv_qasm[key][1]
        if (diff0 > threshold or diff1 > threshold):
            print("found difference: mps = " + str(sv_mps[key]) + " qasm = " + str(sv_qasm[key]))
            failure = 1
            
        if (failure == 1):
            print("test failed")
        else:
            print("test passed")



