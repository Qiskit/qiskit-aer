# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Integration Tests for jump/mark instructions
"""
from ddt import ddt, data
import unittest
import numpy
import logging
from test.terra.backends.simulator_test_case import (
    SimulatorTestCase, supported_methods)
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, Qubit, QuantumRegister, ClassicalRegister
from qiskit.circuit.controlflow import *
from qiskit_aer.library.default_qubits import default_qubits
from qiskit_aer.library.control_flow_instructions import AerMark, AerJump

@ddt
class TestControlFlow(SimulatorTestCase):
    """Test instructions for jump and mark instructions and compiler functions."""

    def add_mark(self, circ, name):
        """Create a mark instruction which can be a destination of jump instructions.
    
        Args:
            name (str): an unique name of this mark instruction in a circuit
        """
        qubits = default_qubits(circ)
        instr = AerMark(name,
                        len(qubits))
        return circ.append(instr, qubits)

    def add_jump(self, circ, jump_to, clbit=None, value=0):
        """Create a jump instruction to move a program counter to a named mark.
    
        Args:
            jump_to (str): a name of a destination mark instruction
            clbit (Clbit): a classical bit for a condition
            value (int): an int value for a condition. if clbit is value, jump is performed.
        """
        qubits = default_qubits(circ)
        instr = AerJump(jump_to, len(qubits))
        if clbit:
            instr.c_if(clbit, value)
        return circ.append(instr, qubits)


    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_jump_always(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(4)
        mark = 'mark'
        self.add_jump(circ, mark)

        for i in range(4):
            circ.h(i)

        self.add_mark(circ, mark)
        
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('0000', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_jump_conditional(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(4, 1)
        mark = 'mark'
        self.add_jump(circ, mark, circ.clbits[0])

        for i in range(4):
            circ.h(i)

        self.add_mark(circ, mark)
        
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('0000 0', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_no_jump_conditional(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(4, 1)
        mark = 'mark'
        self.add_jump(circ, mark, circ.clbits[0], 1)

        for i in range(4):
            circ.h(i)

        self.add_mark(circ, mark)
        
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertNotEqual(len(counts), 1)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_invalid_jump(self, method):
        logging.disable(level=logging.WARN)

        backend = self.backend(method=method)

        circ = QuantumCircuit(4, 1)
        mark = 'mark'
        invalid_mark = 'invalid_mark'
        self.add_jump(circ, invalid_mark, circ.clbits[0])

        for i in range(4):
            circ.h(i)

        self.add_mark(circ, mark)
        
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertNotSuccess(result)

        logging.disable(level=logging.NOTSET)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_duplicated_mark(self, method):
        logging.disable(level=logging.WARN)

        backend = self.backend(method=method)

        circ = QuantumCircuit(4, 1)
        mark = 'mark'
        self.add_jump(circ, mark, circ.clbits[0])

        for i in range(4):
            circ.h(i)

        self.add_mark(circ, mark)
        self.add_mark(circ, mark)
        
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertNotSuccess(result)

        logging.disable(level=logging.NOTSET)


    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_if_true_body_builder(self, method):
        backend = self.backend(method=method)

        qreg = QuantumRegister(4)
        creg = ClassicalRegister(1)
        circ = QuantumCircuit(qreg, creg)
        circ.y(0)
        circ.h(circ.qubits[1:4])
        circ.barrier()
        circ.measure(0, 0)
        
        with circ.if_test((creg, 1)):
            circ.h(circ.qubits[1:4])
        
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('0001 1', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_if_else_body_builder(self, method):
        backend = self.backend(method=method)

        qreg = QuantumRegister(4)
        creg = ClassicalRegister(1)
        circ = QuantumCircuit(qreg, creg)
        circ.h(circ.qubits[1:4])
        circ.barrier()
        circ.measure(0, 0)
        
        with circ.if_test((creg, 1)) as else_:
            pass
        with else_:
            circ.h(circ.qubits[1:4])

        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('0000 0', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_for_loop_builder(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(5, 0)
        
        with circ.for_loop(range(0)) as a:
            circ.ry(a * numpy.pi, 0)
        with circ.for_loop(range(1)) as a:
            circ.ry(a * numpy.pi, 1)
        with circ.for_loop(range(2)) as a:
            circ.ry(a * numpy.pi, 2)
        with circ.for_loop(range(3)) as a:
            circ.ry(a * numpy.pi, 3)
        with circ.for_loop(range(4)) as a:
            circ.ry(a * numpy.pi, 4)

        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('01100', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_for_loop_builder_no_loop_variable(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(5, 0)

        with circ.for_loop(range(0)):
            circ.x(0)
        with circ.for_loop(range(1)):
            circ.x(1)
        with circ.for_loop(range(2)):
            circ.x(2)
        with circ.for_loop(range(3)):
            circ.x(3)
        with circ.for_loop(range(4)):
            circ.x(4)

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('01010', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_for_loop_break_builder(self, method):
        backend = self.backend(method=method)

        qreg = QuantumRegister(5)
        creg = ClassicalRegister(1)
        circ = QuantumCircuit(qreg, creg)
        
        with circ.for_loop(range(0)) as a:
            circ.ry(a * numpy.pi, 0)
            circ.measure(0, 0)
            with circ.if_test((creg, 1)):
                circ.break_loop()
        with circ.for_loop(range(1)) as a:
            circ.ry(a * numpy.pi, 1)
            circ.measure(1, 0)
            with circ.if_test((creg, 1)):
                circ.break_loop()
        with circ.for_loop(range(2)) as a:
            circ.ry(a * numpy.pi, 2)
            circ.measure(2, 0)
            with circ.if_test((creg, 1)):
                circ.break_loop()
        with circ.for_loop(range(3)) as a:
            circ.ry(a * numpy.pi, 3)
            circ.measure(3, 0)
            with circ.if_test((creg, 1)):
                circ.break_loop()
        with circ.for_loop(range(4)) as a:
            circ.ry(a * numpy.pi, 4)
            circ.measure(4, 0)
            with circ.if_test((creg, 1)):
                circ.break_loop()

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('11100 1', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_for_loop_continue_builder(self, method):
        backend = self.backend(method=method)

        qreg = QuantumRegister(5)
        cregs = [ClassicalRegister(1) for _ in range(5)]
        circ = QuantumCircuit(qreg, *cregs)

        with circ.for_loop(range(0)) as a:
            circ.ry(a * numpy.pi, 0)  # dead code
            circ.measure(0, 0)  # dead code
            with circ.if_test((cregs[0], 1)):
                circ.continue_loop()  # dead code
            circ.y(0)  # dead code
            # 1st cbit -> 0
            # 1st meas cbit -> 0

        with circ.for_loop(range(1)) as a:
            circ.ry(a * numpy.pi, 1)
            circ.measure(1, 1)
            with circ.if_test((cregs[1], 1)):
                circ.continue_loop()  # dead code
            circ.y(1)
            # 2nd cbit -> 0
            # 2nd meas cbit -> 1

        with circ.for_loop(range(2)) as a:
            circ.ry(a * numpy.pi, 2)
            circ.measure(2, 2)
            with circ.if_test((cregs[2], 1)):
                circ.continue_loop()
            circ.y(2)
            # 3rd cbit -> 0
            # 3rd meas cbit -> 1

        with circ.for_loop(range(3)) as a:
            circ.ry(a * numpy.pi, 3)
            circ.measure(3, 3)
            with circ.if_test((cregs[3], 1)):
                circ.continue_loop()
            circ.y(3)
            # 4th cbit -> 1
            # 4th meas cbit -> 1

        with circ.for_loop(range(4)) as a:
            circ.ry(a * numpy.pi, 4)
            circ.measure(4, 4)
            with circ.if_test((cregs[4], 1)):
                circ.continue_loop()
            circ.y(4)
            # 5th cbit -> 0
            # 5th meas cbit -> 1

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('11110 0 1 0 0 0', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_while_loop_no_iteration(self, method):
        backend = self.backend(method=method)
    
        qreg = QuantumRegister(1)
        creg = ClassicalRegister(1)
        circ = QuantumCircuit(qreg, creg)
        circ.measure(0, 0)
        with circ.while_loop((creg, 1)):
            circ.y(0)
        circ.measure_all()
    
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
    
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('0 0', counts)
    
    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_while_loop_single_iteration(self, method):
        backend = self.backend(method=method)
    
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(1)
        circ = QuantumCircuit(qreg, creg)
        circ.y(0)
        circ.measure(0, 0)
        
        # does not work
        # while circ.while_loop((creg, 1)):
        #     circ.y(0)
        #     circ.measure(0, 0)
        #     circ.y(1)
        
        circ_while = QuantumCircuit(qreg, creg)
        circ_while.y(0)
        circ_while.measure(0, 0)
        circ_while.y(1)
        circ.while_loop((creg, 1), circ_while, [0, 1], [0])
        
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
    
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('10 0', counts)
    
    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_while_loop_double_iterations(self, method):
        backend = self.backend(method=method)
    
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(1)
        circ = QuantumCircuit(qreg, creg)
        circ.y(0)
        circ.measure(0, 0)
        
        # does not work
        # while circ.while_loop((creg, 1)):
        #    circ.y(0)
        #    circ.measure(0, 0)
        #    circ.y(1)
        
        circ_while = QuantumCircuit(qreg, creg)
        circ_while.measure(0, 0)
        circ_while.y(0)
        circ_while.y(1)
        circ.while_loop((creg, 1), circ_while, [0, 1], [0])

        circ.measure_all()
    
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
    
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('01 0', counts)
    
    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_while_loop_continue(self, method):
        backend = self.backend(method=method)
    
        qreg = QuantumRegister(1)
        creg = ClassicalRegister(1)
        circ = QuantumCircuit(qreg, creg)
        circ.y(0)
        circ.measure(0, 0)
        
        # does not work
        # while circ.while_loop((creg, 1)):
        #    circ.y(0)
        #    circ.measure(0, 0)
        #    circ.continue_loop()
        #    circ.y(0)
        
        circ_while = QuantumCircuit(qreg, creg)
        circ_while.y(0)
        circ_while.measure(0, 0)
        circ_while.continue_loop()
        circ_while.y(0)
        circ_while.break_loop()
        circ.while_loop((creg, 1), circ_while, [0], [0])
        
        circ.measure_all()
    
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
    
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('0 0', counts)
    
    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_nested_loop(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(3)

        with circ.for_loop(range(2)) as a:
            with circ.for_loop(range(2)) as b:
                circ.ry(a * b * numpy.pi, 0)

        with circ.for_loop(range(3)) as a:
            with circ.for_loop(range(3)) as b:
                circ.ry(a * b * numpy.pi, 1)

        with circ.for_loop(range(4)) as a:
            with circ.for_loop(range(2)) as b:
                circ.ry(a * b * numpy.pi, 2)

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
    
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('011', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_while_loop_last(self, method):
        backend = self.backend(method=method)
    
        circ = QuantumCircuit(1, 1)
        circ.h(0)
        circ.measure(0, 0)
        with circ.while_loop((circ.clbits[0], True)):
            circ.h(0)
            circ.measure(0, 0)
    
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)

    @data("statevector", "density_matrix", "matrix_product_state")
    def test_no_invalid_nested_reordering(self, method):
        """Test that the jump/mark system doesn't allow nested conditional marks to jump incorrectly
        relative to their outer marks.  Regression test of gh-1665."""
        backend = self.backend(method=method)

        circuit = QuantumCircuit(3, 3)
        circuit.initialize('010', circuit.qubits)
        circuit.measure(0, 0)
        circuit.measure(1, 1)
        with circuit.if_test((0, True)):
            with circuit.if_test((1, False)):
                circuit.x(2)
        with circuit.if_test((0, False)):
            with circuit.if_test((1, True)):
                circuit.x(2)
        circuit.measure(range(3), range(3))

        result = backend.run(circuit, method=method, shots=100).result()
        self.assertSuccess(result)
        self.assertEqual(result.get_counts(), {"110": 100})

    @data("statevector", "density_matrix", "matrix_product_state")
    def test_no_invalid_reordering_if(self, method):
        """Test that the jump/mark system doesn't allow an unrelated operation to jump inside a
        conditional statement."""
        backend = self.backend(method=method)

        circuit = QuantumCircuit(3, 3)
        circuit.measure(1, 1)
        # This should never be entered.
        with circuit.if_test((1, True)):
            circuit.x(0)
            circuit.x(2)
        # In the topological ordering that `DAGCircuit` uses, this X on qubit 1 will naturally
        # appear between the X on 0 and X on 2 once they are inlined, and the jump/mark instructions
        # won't act as a full barrier, because the if test doesn't touch qubit 1.  In this case, the
        # X on 1 will migrate to between the jump and mark, causing it to be skipped along with the
        # other X gates.  This test ensures that suitable full-width barriers are in place to stop
        # that from happening.
        circuit.x(1)

        circuit.measure(circuit.qubits, circuit.clbits)

        result = backend.run(circuit, method=method, shots=100).result()
        self.assertSuccess(result)
        self.assertEqual(result.get_counts(), {"010": 100})

    @data("statevector", "density_matrix", "matrix_product_state")
    def test_no_invalid_reordering_while(self, method):
        """Test that the jump/mark system doesn't allow an unrelated operation to jump inside a
        conditional statement."""
        backend = self.backend(method=method)

        circuit = QuantumCircuit(3, 3)
        circuit.measure(1, 1)
        # This should never be entered.
        with circuit.while_loop((1, True)):
            circuit.x(0)
            circuit.x(2)
        # In the topological ordering that `DAGCircuit` uses, this X on qubit 1 will naturally
        # appear between the X on 0 and X on 2 once they are inlined, and the jump/mark instructions
        # won't act as a full barrier, because the if test doesn't touch qubit 1.  In this case, the
        # X on 1 will migrate to between the jump and mark, causing it to be skipped along with the
        # other X gates.  This test ensures that suitable full-width barriers are in place to stop
        # that from happening.
        circuit.x(1)

        circuit.measure(circuit.qubits, circuit.clbits)

        result = backend.run(circuit, method=method, shots=100).result()
        self.assertSuccess(result)
        self.assertEqual(result.get_counts(), {"010": 100})
