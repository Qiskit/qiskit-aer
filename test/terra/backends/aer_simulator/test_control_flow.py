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
from qiskit.providers.aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, Qubit, QuantumRegister, ClassicalRegister
from qiskit.circuit.controlflow import *
from qiskit.providers.aer.library.default_qubits import default_qubits
from qiskit.providers.aer.library.control_flow_instructions import AerMark, AerJump

import ast
from types import FunctionType
from typing import Any, List

import inspect
import astunparse
from textwrap import dedent

def dynamic_circuit(func):
    def decorator(*args, **kwargs):
        circ_builder = CircuitBuilder()
        return circ_builder.build(func, *args, **kwargs)(*args, **kwargs)
    return decorator

class CircuitBuilder:

    def __init__(self):
        self._last_circ_id = -1
        self._orig_circ_name = None
        self._circ_name = None
        self._last_clbits = 0
        self._last_param_index = 0
        
    def _new_circuit_id(self):
        self._last_circ_id += 1
        return self._last_circ_id
    
    def _new_parameter_index(self):
        self._last_param_index += 1
        return self._last_param_index
    
    def _type(self, node):
        if isinstance(node, ast.Name):
            return eval(node.id)
        elif isinstance(node, ast.Attribute):
            return eval(node.attr)
        elif isinstance(node, ast.Subscript):
            if self._type(node.value) == List:
                return [self._type(node.slice.value)]
            raise ValueError(f'forbidden type: {node.value}')
        else:
            raise ValueError(f'forbidden type class: {node.__class__}')
    
    def _argument_type(self, arg):
        if not arg.annotation:
            raise ValueError(f'no type information in argument: ' +
                             f'func={function_def.name}, arg={arg_name}, line={arg.lineno}')
        return self._type(arg.annotation)

    def _new_subcircuit_node(self):
        return ast.parse(f'{self._circ_name} = QuantumCircuit(' + 
                         f'*{self._orig_circ_name}.qregs, ' +
                         f'*{self._orig_circ_name}.cregs)'
                         ).body[0]
    
    def _build_test_tuple(self, test):
        if isinstance(test, (ast.Name, ast.Subscript)):
            return ast.parse(f'({astunparse.unparse(test)}, True)')
        elif isinstance(test, ast.Compare):
            lhs = test.left
            if len(test.comparators) != 1 or len(test.ops) != 1:
                raise ValueError(f'forbidden test: {astunparse.unparse(test)}, {ast.dump(test)}')
            rhs = test.comparators[0]
            op = test.ops[0]
            if not isinstance(op, ast.Eq):
                raise ValueError(f'forbidden test: {astunparse.unparse(test)}, {ast.dump(test)}')
            if isinstance(lhs, (ast.Name, ast.Subscript)):
                if isinstance(rhs, ast.Constant):
                    return ast.parse(f'({astunparse.unparse(lhs)}, {astunparse.unparse(rhs)})')
            elif isinstance(lhs, ast.Constant):
                if isinstance(rhs, (ast.Name, ast.Subscript)):
                    return ast.parse(f'({astunparse.unparse(rhs)}, {astunparse.unparse(lhs)})')
            
        raise ValueError(f'forbidden test: {astunparse.unparse(test)}, {ast.dump(test)}')

    def _build_if(self, new_body, node):
        test_tuple = self._build_test_tuple(node.test)
        
        parent_circ_name = self._circ_name
        
        this_circ_name_true = '_circ' + str(self._new_circuit_id())
        self._circ_name = this_circ_name_true
        new_body.append(self._new_subcircuit_node())
        
        self._build_block(new_body, node.body)
        
        if node.orelse:
            this_circ_name_else = '_circ' + str(self._new_circuit_id())
            self._circ_name = this_circ_name_else
            new_body.append(self._new_subcircuit_node())
            
            self._build_block(new_body, node.orelse)

            new_body.append(ast.parse(f'{parent_circ_name}.append(' + 
                                      f'IfElseOp({astunparse.unparse(test_tuple)}, {this_circ_name_true}, {this_circ_name_else}), ' + 
                                      f'range(len({parent_circ_name}.qubits)), ' + 
                                      f'range(len({parent_circ_name}.clbits)))'))
        else:
            new_body.append(ast.parse(f'{parent_circ_name}.append(' + 
                                      f'IfElseOp({astunparse.unparse(test_tuple)}, {this_circ_name_true}), ' + 
                                      f'range(len({parent_circ_name}.qubits)), ' + 
                                      f'range(len({parent_circ_name}.clbits)))'))
        
        self._circ_name = parent_circ_name
    
    def _build_while(self, new_body, node):
        test_tuple = self._build_test_tuple(node.test)
        
        parent_circ_name = self._circ_name
        
        this_circ_name = '_circ' + str(self._new_circuit_id())
        self._circ_name = this_circ_name
        new_body.append(self._new_subcircuit_node())
        
        self._build_block(new_body, node.body)
        
        new_body.append(ast.parse(f'{parent_circ_name}.append(' + 
                                  f'WhileLoopOp({astunparse.unparse(test_tuple)}, {this_circ_name}), ' + 
                                  f'range(len({parent_circ_name}.qubits)), ' + 
                                  f'range(len({parent_circ_name}.clbits)))'))
        
        self._circ_name = parent_circ_name

    def _build_call_expr(self, new_body, node):
        call = node.value
        if (isinstance(call.func, ast.Attribute) and
            isinstance(call.func.value, ast.Name) and
            call.func.value.id == self._orig_circ_name):
            call.func.value.id = self._circ_name
        
        new_body.append(node)
    
    def _build_for(self, new_body, node):
        if not isinstance(node.target, ast.Name):
            raise ValueError(f'no parameter in a loop: {astunparse.unparse(node)}')
        
        param_name = node.target.id
        param_name_id = param_name + str(self._new_parameter_index())
        new_body.append(ast.parse(f"{param_name} = Parameter('{param_name_id}')"))
        
        parent_circ_name = self._circ_name
        this_circ_name = '_circ' + str(self._new_circuit_id())
        self._circ_name = this_circ_name
        new_body.append(self._new_subcircuit_node())
        
        self._build_block(new_body, node.body)
        
        new_body.append(ast.parse(f'{parent_circ_name}.append(' + 
                                      f'ForLoopOp({astunparse.unparse(node.iter)}, {param_name}, {this_circ_name}), ' + 
                                      f'range(len({parent_circ_name}.qubits)), ' + 
                                      f'range(len({parent_circ_name}.clbits)))'))
        
        self._circ_name = parent_circ_name
        
    def _build_break(self, new_body, node):
        new_body.append(ast.parse(f'{self._circ_name}.append(' + 
                                  f'BreakLoopOp(len({self._circ_name}.qubits), len({self._circ_name}.clbits)), ' +
                                  f'range(len({self._circ_name}.qubits)), ' + 
                                  f'range(len({self._circ_name}.clbits)))'))

    def _build_continue(self, new_body, node):
        new_body.append(ast.parse(f'{self._circ_name}.append(' + 
                                  f'ContinueLoopOp(len({self._circ_name}.qubits), len({self._circ_name}.clbits)), ' +
                                  f'range(len({self._circ_name}.qubits)), ' + 
                                  f'range(len({self._circ_name}.clbits)))'))
        
    def _build_pass(self, new_body, node):
        pass

    def _build_block(self, new_body, node):
        for child in node:
            if isinstance(child, ast.If):
                self._build_if(new_body, child)
            elif isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
                self._build_call_expr(new_body, child)
            elif isinstance(child, ast.For):
                self._build_for(new_body, child)
            elif isinstance(child, ast.Break):
                self._build_break(new_body, child)
            elif isinstance(child, ast.Continue):
                self._build_continue(new_body, child)
            elif isinstance(child, ast.While):
                self._build_while(new_body, child)
            elif isinstance(child, ast.Pass):
                self._build_pass(new_body, child)
            else:
                print(child.__class__)
                raise ValueError(f'forbidden code: {astunparse.unparse(child)}')

    def _build_code(self, function: FunctionType, *args: Any, **kwargs: Any):
        function_ast = ast.parse(dedent(inspect.getsource(function)))
        
        function_def = function_ast.body[0]
        
        arguments = { arg.arg: self._argument_type(arg) for arg in function_def.args.args }
        
        if len(arguments) == 0 or arguments[function_def.args.args[0].arg] != QuantumCircuit:
            raise ValueError('dynamic circuit requires QuantumCricuit as the 1st argument')

        if len(function_def.body) == 0:
            raise ValueError('dynamic circuit requires at least one statement')
        
        qubit_defs = []
        qubits_defs = []
        
        for k, v in arguments.items():
            if v == Qubit:
                qubit_defs.append(k)
            elif isinstance(v, list) and v[0] == Qubit:
                qubits_defs.append(k)
        
        self._orig_circ_name = function_def.args.args[0].arg
        this_circ_name = '_circ' + str(self._new_circuit_id())
        self._circ_name = this_circ_name
        
        new_body = []
        
        # hedder
        new_body.append(ast.parse('import numpy').body[0])
        new_body.append(ast.parse('from qiskit.circuit import QuantumCircuit, Parameter').body[0])
        new_body.append(ast.parse('from qiskit.circuit.controlflow import WhileLoopOp, ForLoopOp, IfElseOp, BreakLoopOp, ContinueLoopOp').body[0])
        
        # body
        new_body.append(self._new_subcircuit_node())
        self._build_block(new_body, function_def.body)
        
        # footer
        new_body.append(ast.parse(f'{self._orig_circ_name}.compose({self._circ_name}, ' +
                                  f'range({self._circ_name}.num_qubits), ' + 
                                  f'range({self._circ_name}.num_clbits), ' + 
                                  'inplace=True)'))
        
        function_def.body = new_body
        function_def.name = '_____' + function_def.name
        function_def.decorator_list.clear()
        
        # print(astunparse.unparse(function_ast))
        return function_def.name, astunparse.unparse(function_ast)

    def build(self, function: FunctionType, *args: Any, **kwargs: Any):
        function_name, function_code = self._build_code(function, *args, **kwargs)
        exec(function_code)
        return eval(function_name)


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
        self.assertSuccess(result, False)

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
        self.assertSuccess(result, False)

        logging.disable(level=logging.NOTSET)


    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_if_true_body(self, method):
        backend = self.backend(method=method)

        @dynamic_circuit
        def test_if_true_body_circuit(circ: QuantumCircuit):
            circ.y(0)
            circ.h(circ.qubits[1:4])
            circ.barrier()
            circ.measure(0, 0)
        
            if circ.clbits[0]:
                circ.h(circ.qubits[1:4])

        circ = QuantumCircuit(4, 1)
        test_if_true_body_circuit(circ)
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('0001 1', counts)

    @unittest.skip('builder is not ready')
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
    def test_if_else_body(self, method):
        backend = self.backend(method=method)

        @dynamic_circuit
        def test_if_else_body_circuit(circ: QuantumCircuit):
            circ.h(circ.qubits[1:4])
            circ.barrier()
            circ.measure(0, 0)
        
            if circ.clbits[0]:
                pass
            else:
                circ.h(circ.qubits[1:4])

        circ = QuantumCircuit(4, 1)
        test_if_else_body_circuit(circ)
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('0000 0', counts)

    @unittest.skip('builder is not ready')
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
    def test_for_loop(self, method):
        backend = self.backend(method=method)

        @dynamic_circuit
        def test_for_loop_circuit(circ: QuantumCircuit):
            for a in range(0):
                circ.ry(a * numpy.pi, 0)
                
            for a in range(1):
                circ.ry(a * numpy.pi, 1)
        
            for a in range(2):
                circ.ry(a * numpy.pi, 2)
        
            for a in range(3):
                circ.ry(a * numpy.pi, 3)
                
            for a in range(4):
                circ.ry(a * numpy.pi, 4)

        circ = QuantumCircuit(5, 0)
        test_for_loop_circuit(circ)
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('01100', counts)

    @unittest.skip('builder is not ready')
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
    def test_for_loop_break(self, method):
        backend = self.backend(method=method)

        @dynamic_circuit
        def test_for_loop_break_circuit(circ: QuantumCircuit):
            for a in range(0):
                circ.ry(a * numpy.pi, 0)
                circ.measure(0, 0)
                if circ.clbits[0]:
                    break
                
            for a in range(1):
                circ.ry(a * numpy.pi, 1)
                circ.measure(1, 0)
                if circ.clbits[0]:
                    break
        
            for a in range(2):
                circ.ry(a * numpy.pi, 2)
                circ.measure(2, 0)
                if circ.clbits[0]:
                    break
        
            for a in range(3):
                circ.ry(a * numpy.pi, 3)
                circ.measure(3, 0)
                if circ.clbits[0]:
                    break
                
            for a in range(4):
                circ.ry(a * numpy.pi, 4)
                circ.measure(4, 0)
                if circ.clbits[0]:
                    break
        
        circ = QuantumCircuit(5, 1)
        test_for_loop_break_circuit(circ)
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('11100 1', counts)

    @unittest.skip('builder is not ready')
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
    def test_for_loop_continue(self, method):
        backend = self.backend(method=method)

        @dynamic_circuit
        def test_for_loop_continue_circuit(circ: QuantumCircuit):
            for a in range(0):
                circ.ry(a * numpy.pi, 0) # dead code
                circ.measure(0, 0) # dead code
                if circ.clbits[0]: # dead code
                    continue # dead code
                circ.y(0) # dead code
            # 1st cbit -> 0
            # 1st meas cbit -> 0
                
            for a in range(1):
                circ.ry(a * numpy.pi, 1)
                circ.measure(1, 1)
                if circ.clbits[1]:
                    continue # dead code
                circ.y(1)
            # 2nd cbit -> 0
            # 2nd meas cbit -> 1
        
            for a in range(2):
                circ.ry(a * numpy.pi, 2)
                circ.measure(2, 2)
                if circ.clbits[2]:
                    continue # dead code
                circ.y(2)
            # 3rd cbit -> 0
            # 3rd meas cbit -> 1
        
            for a in range(3):
                circ.ry(a * numpy.pi, 3)
                circ.measure(3, 3)
                if circ.clbits[3]:
                    continue
                circ.y(3)
            # 4th cbit -> 1
            # 4th meas cbit -> 1
                
            for a in range(4):
                circ.ry(a * numpy.pi, 4)
                circ.measure(4, 4)
                if circ.clbits[4]:
                    continue
                circ.y(4)
            # 5th cbit -> 0
            # 5th meas cbit -> 1

        circ = QuantumCircuit(5, 5)
        test_for_loop_continue_circuit(circ)
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('11110 01000', counts)

    @unittest.skip('builder is not ready')
    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_for_loop_continue_builder(self, method):
        backend = self.backend(method=method)

        qreg = QuantumRegister(5)
        creg = ClassicalRegister(5)
        circ = QuantumCircuit(qreg, creg)

        with circ.for_loop(range(0)) as a:
            circ.ry(a * numpy.pi, 0)  # dead code
            circ.measure(0, 0)  # dead code
            with circ.if_test((circ.clbits[0], 1)):
                circ.continue_loop()  # dead code
            circ.y(0)  # dead code
            # 1st cbit -> 0
            # 1st meas cbit -> 0

        with circ.for_loop(range(1)) as a:
            circ.ry(a * numpy.pi, 1)
            circ.measure(1, 1)
            with circ.if_test((circ.clbits[1], 1)):
                circ.continue_loop()  # dead code
            circ.y(1)
            # 2nd cbit -> 0
            # 2nd meas cbit -> 1

        with circ.for_loop(range(2)) as a:
            circ.ry(a * numpy.pi, 2)
            circ.measure(2, 2)
            with circ.if_test((circ.clbits[2], 1)):
                circ.continue_loop()
            circ.y(2)
            # 3rd cbit -> 0
            # 3rd meas cbit -> 1

        with circ.for_loop(range(3)) as a:
            circ.ry(a * numpy.pi, 3)
            circ.measure(3, 3)
            with circ.if_test((circ.clbits[3], 1)):
                circ.continue_loop()
            circ.y(3)
            # 4th cbit -> 1
            # 4th meas cbit -> 1

        with circ.for_loop(range(4)) as a:
            circ.ry(a * numpy.pi, 4)
            circ.measure(4, 4)
            with circ.if_test((circ.clbits[4], 1)):
                circ.continue_loop()
            circ.y(4)
            # 5th cbit -> 0
            # 5th meas cbit -> 1

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('11110 01000', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_while_loop_no_iteration(self, method):
        backend = self.backend(method=method)

        @dynamic_circuit
        def test_while_loop_no_iteration_circuit(circ: QuantumCircuit):
            circ.measure(0, 0)
            while circ.clbits[0]:
                circ.y(0)
                break
        
        circ = QuantumCircuit(1, 1)
        test_while_loop_no_iteration_circuit(circ)
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('0 0', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_while_loop_single_iteration(self, method):
        backend = self.backend(method=method)

        @dynamic_circuit
        def test_while_loop_single_iteration_circuit(circ: QuantumCircuit):
            circ.y(0)
            circ.measure(0, 0)
            while circ.clbits[0]:
                circ.y(0)
                circ.measure(0, 0)
                circ.y(1)
            
        circ = QuantumCircuit(2, 1)
        test_while_loop_single_iteration_circuit(circ)
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('10 0', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_while_loop_double_iterations(self, method):
        backend = self.backend(method=method)

        @dynamic_circuit
        def test_while_loop_double_iterations_circuit(circ: QuantumCircuit):
            circ.measure(0, 0)
            circ.y(0)
            while circ.clbits[0]:
                circ.measure(0, 0)
                circ.y(0)
                circ.y(1)
        
        circ = QuantumCircuit(2, 1)
        test_while_loop_double_iterations_circuit(circ)
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('01 0', counts)

    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_while_loop_continue(self, method):
        backend = self.backend(method=method)

        @dynamic_circuit
        def test_while_loop_continue_circuit(circ: QuantumCircuit):
            circ.y(0)
            circ.measure(0, 0)
            while circ.clbits[0]:
                circ.y(0)
                circ.measure(0, 0)
                continue
                circ.y(0) # must be dead code
                break
    
        circ = QuantumCircuit(1, 1)
        test_while_loop_continue_circuit(circ)
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('0 0', counts)


    @data('statevector', 'density_matrix', 'matrix_product_state')
    def test_nested_loop(self, method):
        backend = self.backend(method=method)

        @dynamic_circuit
        def test_for_loop_circuit(circ: QuantumCircuit):
            for a in range(2):
                for b in range(2):
                    circ.ry(a * b * numpy.pi, 0)
                
            for a in range(3):
                for b in range(3):
                    circ.ry(a * b * numpy.pi, 1)
                    
            for a in range(4):
                for b in range(2):
                    circ.ry(a * b * numpy.pi, 2)

        circ = QuantumCircuit(3, 0)

        test_for_loop_circuit(circ)
        circ.measure_all()
        
        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)
        
        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn('011', counts)
