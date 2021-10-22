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
Integration Tests for AerCompiler and jump/mark instructions
"""
from ddt import ddt, data
import numpy
import logging
from test.terra.backends.simulator_test_case import (
    SimulatorTestCase, supported_methods)
from qiskit.providers.aer import AerSimulator, AerCompiler
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.controlflow import *

@ddt
class TestControlFlow(SimulatorTestCase):
    """Test instructions for jump and mark instructions and compiler functions."""

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_jump_always(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(4)
        mark = 'mark'
        circ.jump(mark)

        for i in range(4):
            circ.h(i)

        circ.mark(mark)
        
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
        circ.jump(mark, circ.clbits[0])

        for i in range(4):
            circ.h(i)

        circ.mark(mark)
        
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
        circ.jump(mark, circ.clbits[0], 1)

        for i in range(4):
            circ.h(i)

        circ.mark(mark)
        
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
        circ.jump(invalid_mark, circ.clbits[0])

        for i in range(4):
            circ.h(i)

        circ.mark(mark)
        
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result, False)

        logging.disable(level=logging.NOTSET)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_duplicated_mark(self, method):
        logging.disable(level=logging.WARN)

        backend = self.backend(method=method)

        circ = QuantumCircuit(4, 1)
        mark = 'mark'
        circ.jump(mark, circ.clbits[0])

        for i in range(4):
            circ.h(i)

        circ.mark(mark)
        circ.mark(mark)
        
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result, False)

        logging.disable(level=logging.NOTSET)


    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_if_true_body(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(4, 1)
        circ.y(0)
        for i in range(1, 4):
            circ.h(i)
        circ.barrier()
        circ.measure(0, 0)

        true_body = QuantumCircuit(3)
        for i in range(3):
            true_body.h(i)
        
        circ.append(IfElseOp(circ.clbits[0], true_body), [1, 2, 3], [])

        circ.measure_all()
        
        circ = transpile(AerCompiler().compile_circuit(circ), backend, optimization_level=0)
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('0001 1', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_if_else_body(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(4, 1)
        for i in range(1, 4):
            circ.h(i)
        circ.barrier()
        circ.measure(0, 0)

        true_body = QuantumCircuit(3)
        
        else_body = QuantumCircuit(3)
        for i in range(3):
            else_body.h(i)
        
        circ.append(IfElseOp(circ.clbits[0], true_body, else_body), [1, 2, 3], [])

        circ.measure_all()
        
        circ = transpile(AerCompiler().compile_circuit(circ), backend, optimization_level=0)
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('0000 0', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_for_loop(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(5)
        
        loop_parameter = Parameter('a')
        
        loop_body = QuantumCircuit(1)
        loop_body.ry(loop_parameter * numpy.pi, 0)
        
        circ.append(ForLoopOp(loop_parameter, range(0), loop_body.copy()), [0], []) # () -> 0
        circ.append(ForLoopOp(loop_parameter, range(1), loop_body.copy()), [1], []) # (pi * 0) -> 0
        circ.append(ForLoopOp(loop_parameter, range(2), loop_body.copy()), [2], []) # (pi * 0) + (pi * 1) -> 1
        circ.append(ForLoopOp(loop_parameter, range(3), loop_body.copy()), [3], []) # (pi * 0) + (pi * 1) + (pi * 2) -> 1
        circ.append(ForLoopOp(loop_parameter, range(4), loop_body.copy()), [3], []) # (pi * 0) + (pi * 1) + (pi * 2) + (pi * 3) -> 0
        circ.measure_all()
        
        circ = transpile(AerCompiler().compile_circuit(circ), backend, optimization_level=0)
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('01100', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_for_loop_break(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(5, 1)
        
        loop_parameter = Parameter('a')
        
        loop_body = QuantumCircuit(1, 1)
        loop_body.ry(loop_parameter * numpy.pi, 0)
        loop_body.measure(0, 0)
        
        true_body = QuantumCircuit(1)
        true_body.append(BreakLoopOp(1, 0), [0], [])
        loop_body.append(IfElseOp(loop_body.clbits[0], true_body), [0], [])
        
        circ.append(ForLoopOp(loop_parameter, range(0), loop_body.copy()), [0], [0]) # () -> 0
        circ.append(ForLoopOp(loop_parameter, range(1), loop_body.copy()), [1], [0]) # (pi * 0) -> 0
        circ.append(ForLoopOp(loop_parameter, range(2), loop_body.copy()), [2], [0]) # (pi * 0) + (pi * 1) -> 1
        circ.append(ForLoopOp(loop_parameter, range(3), loop_body.copy()), [3], [0]) # (pi * 0) + (pi * 1) break -> 1
        circ.append(ForLoopOp(loop_parameter, range(4), loop_body.copy()), [4], [0]) # (pi * 0) + (pi * 1) break -> 1
        circ.measure_all()
        
        circ = transpile(AerCompiler().compile_circuit(circ), backend, optimization_level=0)
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('11100 1', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_for_loop_continue(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(4, 4)
        
        loop_parameter = Parameter('a')
        
        loop_body = QuantumCircuit(1, 1)
        loop_body.ry(loop_parameter * numpy.pi, 0)
        loop_body.measure(0, 0)
        
        true_body = QuantumCircuit(1)
        true_body.append(ContinueLoopOp(1, 0), [0], [])
        loop_body.append(IfElseOp(loop_body.clbits[0], true_body), [0], [])
        
        loop_body.y(0)
        
        circ.append(ForLoopOp(loop_parameter, range(0), loop_body.copy()), [0], [0]) # () -> 0
        circ.append(ForLoopOp(loop_parameter, range(1), loop_body.copy()), [1], [1]) # ((pi * 0) + (pi)) -> 1
        circ.append(ForLoopOp(loop_parameter, range(2), loop_body.copy()), [2], [2]) # ((pi * 0) + (pi)) + ((pi * 1) + pi) -> 1
        circ.append(ForLoopOp(loop_parameter, range(3), loop_body.copy()), [3], [3]) # ((pi * 0) + (pi)) + ((pi * 1) + pi) + ((pi * 2) + continue) -> 1
        circ.measure_all()
        
        circ = transpile(AerCompiler().compile_circuit(circ), backend, optimization_level=0)
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('1110 1000', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_while_loop_no_iteration(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(1, 1)
        
        loop_body = QuantumCircuit(1, 1)
        loop_body.y(0)

        circ.append(WhileLoopOp(circ.clbits[0], loop_body), [0], [0])        
        circ.measure_all()
        
        circ = transpile(AerCompiler().compile_circuit(circ), backend, optimization_level=0)
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('0 0', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_while_loop_single_iteration(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(2, 1)
        circ.y(0) # 1st bit -> 1
        circ.measure(0, 0) # 1st bit (creg) -> 1
        
        loop_body = QuantumCircuit(2, 1)
        loop_body.y(1) # 2nd bit -> 1
        loop_body.append(BreakLoopOp(1, 0), [1], [0])

        circ.append(WhileLoopOp(circ.clbits[0], loop_body), [0, 1], [0])
        circ.measure_all()
        
        circ = transpile(AerCompiler().compile_circuit(circ), backend, optimization_level=0)
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('11 1', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_while_loop_double_iterations(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(2, 1)
        circ.y(0) # 1st bit -> 1
        circ.measure(0, 0) # 1st bit (creg) -> 1
        
        loop_body = QuantumCircuit(2, 1)
        loop_body.measure(0, 0) # 1st bit (creg) -> 1
        loop_body.y(0) # 1st bit -> 0 -> 1
        loop_body.y(1) # 2nd bit -> 1 -> 0

        circ.append(WhileLoopOp(circ.clbits[0], loop_body), [0, 1], [0])
        circ.measure_all()
        
        circ = transpile(AerCompiler().compile_circuit(circ), backend, optimization_level=0)
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('01 0', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_while_loop_continue(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(1, 1)
        circ.y(0) # 1st bit -> 1
        circ.measure(0, 0) # 1st bit (creg) -> 1
        
        loop_body = QuantumCircuit(1, 1)
        loop_body.y(0) # 1st bit -> 0
        loop_body.measure(0, 0) # 1st bit (creg) -> 1
        loop_body.append(ContinueLoopOp(1, 0), [0], [])
        loop_body.y(0) # must be dead code. if this is performed, 1st bit -> 1
        loop_body.append(BreakLoopOp(1, 0), [0], [])

        circ.append(WhileLoopOp(circ.clbits[0], loop_body), [0], [0])
        circ.measure_all()
        
        circ = transpile(AerCompiler().compile_circuit(circ), backend, optimization_level=0)
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('0 0', counts)