# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
QasmSimulator Integration Tests
"""

from test.terra.utils import common
from qiskit import compile, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer import QasmSimulator
from test.terra.utils import ref_1q_clifford
from test.terra.utils import ref_2q_clifford
import json


class QasmFusionTests(common.QiskitAerTestCase):
    """QasmSimulator fusion tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {'fusion_verbose': True}
    
    def create_statevector_circuit(self):
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        circuit = QuantumCircuit(qr, cr)
        circuit.u3(0.1,0.1,0.1,qr[0])
        circuit.barrier(qr)
        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.u3(0.1,0.1,0.1,qr[0])
        circuit.barrier(qr)
        circuit.measure(qr, cr)
        return circuit
    
    def test_clifford_no_fusion(self):
        """Test Fusion with clifford simulator"""
        shots = 100
        circuits = ref_2q_clifford.cx_gate_circuits_deterministic(final_measure=True)
        qobj = compile(circuits, self.SIMULATOR, shots=shots)
        result = self.SIMULATOR.run(qobj, backend_options={'fusion_verbose': True}).result()
        self.is_completed(result)
        self.assertTrue('results' in result.as_dict(), 
                        msg="results must exist in result")
        self.assertTrue('metadata' in result.as_dict()['results'][0], 
                        msg="metadata must exist in results[0]")
        self.assertTrue('fusion_verbose' not in result.as_dict()['results'][0]['metadata'], 
                        msg="fusion must not work for clifford")

    def test_fusion_verbose(self):
        """Test Fusion with verbose option"""
        circuit = self.create_statevector_circuit()
        
        shots = 100
        qobj = compile([circuit], self.SIMULATOR, shots=shots, seed=1)
        
        result_verbose = self.SIMULATOR.run(qobj, backend_options={'fusion_enable': True, 'fusion_verbose': True}).result()
        self.is_completed(result_verbose)
        self.assertTrue('results' in result_verbose.as_dict(), 
                        msg="results must exist in result")
        self.assertTrue('metadata' in result_verbose.as_dict()['results'][0], 
                        msg="metadata must exist in results[0]")
        self.assertTrue('fusion_verbose' in result_verbose.as_dict()['results'][0]['metadata'], 
                        msg="fusion must work for satevector")

        result_nonverbose = self.SIMULATOR.run(qobj, backend_options={'fusion_enable': True, 'fusion_verbose': False}).result()
        self.is_completed(result_nonverbose)
        self.assertTrue('results' in result_nonverbose.as_dict(), 
                        msg="results must exist in result")
        self.assertTrue('metadata' in result_nonverbose.as_dict()['results'][0], 
                        msg="metadata must exist in results[0]")
        self.assertTrue('fusion_verbose' not in result_nonverbose.as_dict()['results'][0]['metadata'], 
                        msg="verbose must not work if fusion_verbose is False")

        result_default = self.SIMULATOR.run(qobj, backend_options={'fusion_enable': True}).result()
        self.is_completed(result_default)
        self.assertTrue('results' in result_default.as_dict(), 
                        msg="results must exist in result")
        self.assertTrue('metadata' in result_default.as_dict()['results'][0], 
                        msg="metadata must exist in results[0]")
        self.assertTrue('fusion_verbose' not in result_default.as_dict()['results'][0]['metadata'], 
                        msg="verbose must not work if fusion_verbose is False")
        
    def test_control_fusion(self):
        """Test Fusion enable/disable option"""
        shots = 100
        circuit = self.create_statevector_circuit()
        qobj = compile([circuit], self.SIMULATOR, shots=shots, seed=0)
        
        result_verbose = self.SIMULATOR.run(qobj, backend_options={'fusion_enable': True, 'fusion_verbose': True}).result()
        self.is_completed(result_verbose)
        self.assertTrue('results' in result_verbose.as_dict(), 
                        msg="results must exist in result")
        self.assertTrue('metadata' in result_verbose.as_dict()['results'][0], 
                        msg="metadata must exist in results[0]")
        self.assertTrue('fusion_verbose' in result_verbose.as_dict()['results'][0]['metadata'], 
                        msg="fusion must work for satevector")

        result_disabled = self.SIMULATOR.run(qobj, backend_options={'fusion_enable': False, 'fusion_verbose': True}).result()
        self.is_completed(result_disabled)
        self.assertTrue('results' in result_disabled.as_dict(), 
                        msg="results must exist in result")
        self.assertTrue('metadata' in result_disabled.as_dict()['results'][0], 
                        msg="metadata must exist in results[0]")
        self.assertTrue('fusion_verbose' not in result_disabled.as_dict()['results'][0]['metadata'], 
                        msg="fusion must not work with fusion_enable is False")

        result_default = self.SIMULATOR.run(qobj, backend_options={'fusion_verbose': True}).result()
        self.is_completed(result_default)
        self.assertTrue('results' in result_default.as_dict(), 
                        msg="results must exist in result")
        self.assertTrue('metadata' in result_default.as_dict()['results'][0], 
                        msg="metadata must exist in results[0]")
        self.assertTrue('fusion_verbose' not in result_default.as_dict()['results'][0]['metadata'], 
                        msg="fusion must not work by default for satevector")


    def test_fusion_operations(self):
        """Test Fusion enable/disable option"""
        shots = 100
        
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        circuit = QuantumCircuit(qr, cr)
        
        for i in range(10):
            circuit.h(qr[i])
            circuit.barrier(qr)
            
        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.cx(qr[2], qr[3])
        circuit.barrier(qr)
        circuit.u3(0.1,0.1,0.1,qr[3])
        circuit.barrier(qr)
        circuit.u3(0.1,0.1,0.1,qr[3])
        circuit.barrier(qr)
        
        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.cx(qr[2], qr[3])
        circuit.barrier(qr)
        circuit.u3(0.1,0.1,0.1,qr[3])
        circuit.barrier(qr)
        circuit.u3(0.1,0.1,0.1,qr[3])
        circuit.barrier(qr)
        
        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.x(qr[0])
        circuit.barrier(qr)
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.cx(qr[2], qr[3])
        circuit.barrier(qr)
        circuit.u3(0.1,0.1,0.1,qr[3])
        circuit.barrier(qr)
        circuit.u3(0.1,0.1,0.1,qr[3])
        circuit.barrier(qr)
        
        circuit.measure(qr, cr)

        qobj = compile([circuit], self.SIMULATOR, shots=shots, seed=1)
 
        result_fusion = self.SIMULATOR.run(qobj, backend_options={'fusion_enable': True, 'fusion_verbose': True}).result()
        self.is_completed(result_fusion)
        
        result_nonfusion = self.SIMULATOR.run(qobj, backend_options={'fusion_enable': False, 'fusion_verbose': True}).result()
        self.is_completed(result_nonfusion)

        self.assertDictAlmostEqual(result_fusion.get_counts(circuit), result_nonfusion.get_counts(circuit), delta=0.0, msg="fusion x-x-x was failed")
      