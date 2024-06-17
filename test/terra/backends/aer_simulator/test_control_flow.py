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
from test.terra.backends.simulator_test_case import SimulatorTestCase, supported_methods
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, Qubit, Clbit, QuantumRegister, ClassicalRegister
from qiskit.circuit.controlflow import *
from qiskit.circuit.classical import expr, types
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
        instr = AerMark(name, len(qubits))
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

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_jump_always(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(4)
        mark = "mark"
        self.add_jump(circ, mark)

        for i in range(4):
            circ.h(i)

        self.add_mark(circ, mark)

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("0000", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_jump_conditional(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(4, 1)
        mark = "mark"
        self.add_jump(circ, mark, circ.clbits[0])

        for i in range(4):
            circ.h(i)

        self.add_mark(circ, mark)

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("0000 0", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_no_jump_conditional(self, method):
        backend = self.backend(method=method)

        circ = QuantumCircuit(4, 1)
        mark = "mark"
        self.add_jump(circ, mark, circ.clbits[0], 1)

        for i in range(4):
            circ.h(i)

        self.add_mark(circ, mark)

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        self.assertNotEqual(len(counts), 1)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_invalid_jump(self, method):
        logging.disable(level=logging.WARN)

        backend = self.backend(method=method)

        circ = QuantumCircuit(4, 1)
        mark = "mark"
        invalid_mark = "invalid_mark"
        self.add_jump(circ, invalid_mark, circ.clbits[0])

        for i in range(4):
            circ.h(i)

        self.add_mark(circ, mark)

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertNotSuccess(result)

        logging.disable(level=logging.NOTSET)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_duplicated_mark(self, method):
        logging.disable(level=logging.WARN)

        backend = self.backend(method=method)

        circ = QuantumCircuit(4, 1)
        mark = "mark"
        self.add_jump(circ, mark, circ.clbits[0])

        for i in range(4):
            circ.h(i)

        self.add_mark(circ, mark)
        self.add_mark(circ, mark)

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertNotSuccess(result)

        logging.disable(level=logging.NOTSET)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
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
        self.assertIn("0001 1", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
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
        self.assertIn("0000 0", counts)

    @data("statevector", "density_matrix", "matrix_product_state")
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
        self.assertIn("01100", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
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
        self.assertIn("01010", counts)

    @data("statevector", "density_matrix", "matrix_product_state")
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
        self.assertIn("11100 1", counts)

    @data("statevector", "density_matrix", "matrix_product_state")
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
        self.assertIn("11110 0 1 0 0 0", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
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
        self.assertIn("0 0", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
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
        self.assertIn("10 0", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
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
        self.assertIn("01 0", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
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
        self.assertIn("0 0", counts)

    @data("statevector", "density_matrix", "matrix_product_state")
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
        self.assertIn("011", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
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

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_no_invalid_nested_reordering(self, method):
        """Test that the jump/mark system doesn't allow nested conditional marks to jump incorrectly
        relative to their outer marks.  Regression test of gh-1665."""
        backend = self.backend(method=method)

        circuit = QuantumCircuit(3, 3)
        circuit.initialize("010", circuit.qubits)
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

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
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

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
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

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_transpile_break_and_continue_loop(self, method):
        """Test that transpiler can transpile break_loop and continue_loop with AerSimulator"""

        backend = self.backend(method=method, seed_simulator=1)

        qc = QuantumCircuit(1, 1)

        with qc.for_loop(range(1000)):
            qc.h(0)
            qc.measure(0, 0)
            with qc.if_test((0, True)):
                qc.break_loop()

        transpiled = transpile(qc, backend)
        result = backend.run(transpiled, method=method, shots=100).result()
        self.assertEqual(result.get_counts(), {"1": 100})

        qc = QuantumCircuit(1, 1)

        with qc.for_loop(range(1000)):
            qc.h(0)
            qc.measure(0, 0)
            with qc.if_test((0, False)):
                qc.continue_loop()
            qc.break_loop()

        transpiled = transpile(qc, backend)
        result = backend.run(transpiled, method=method, shots=100).result()
        self.assertEqual(result.get_counts(), {"1": 100})

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_switch_clbit(self, method):
        """Test that a switch statement can be constructed with a bit as a condition."""

        backend = self.backend(method=method)

        qubit = Qubit()
        clbit = Clbit()
        case0 = QuantumCircuit([qubit, clbit])
        case0.x(0)
        case1 = QuantumCircuit([qubit, clbit])
        case1.h(0)

        op = SwitchCaseOp(clbit, [(False, case0), (True, case1)])

        qc0 = QuantumCircuit([qubit, clbit])
        qc0.measure(qubit, clbit)
        qc0.append(op, [qubit], [clbit])
        qc0.measure_all()

        qc0_expected = QuantumCircuit([qubit, clbit])
        qc0_expected.measure(qubit, clbit)
        qc0_expected.append(case0, [qubit], [clbit])
        qc0_expected.measure_all()
        qc0_expected = transpile(qc0_expected, backend)

        ret0 = backend.run(qc0, shots=10000, seed_simulator=1).result()
        ret0_expected = backend.run(qc0_expected, shots=10000, seed_simulator=1).result()
        self.assertSuccess(ret0)
        self.assertEqual(ret0.get_counts(), ret0_expected.get_counts())

        qc1 = QuantumCircuit([qubit, clbit])
        qc1.x(0)
        qc1.measure(qubit, clbit)
        qc1.append(op, [qubit], [clbit])
        qc1.measure_all()

        qc1_expected = QuantumCircuit([qubit, clbit])
        qc1_expected.x(0)
        qc1_expected.measure(qubit, clbit)
        qc1_expected.append(case1, [qubit], [clbit])
        qc1_expected.measure_all()
        qc1_expected = transpile(qc1_expected, backend)

        ret1 = backend.run(qc1, shots=10000, seed_simulator=1).result()
        ret1_expected = backend.run(qc1_expected, shots=10000, seed_simulator=1).result()
        self.assertSuccess(ret1)
        self.assertEqual(ret1.get_counts(), ret1_expected.get_counts())

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_switch_register(self, method):
        """Test that a switch statement can be constructed with a register as a condition."""

        backend = self.backend(method=method, seed_simulator=1)

        qubit0 = Qubit()
        qubit1 = Qubit()
        qubit2 = Qubit()
        creg = ClassicalRegister(2)
        case1 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        case2.x(1)
        case3 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        case3.x(2)

        op = SwitchCaseOp(creg, [(0, case1), (1, case2), (2, case3)])

        qc0 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc0.measure(0, creg[0])
        qc0.append(op, [qubit0, qubit1, qubit2], creg)
        qc0.measure_all()

        ret0 = backend.run(qc0, shots=100).result()
        self.assertSuccess(ret0)
        self.assertEqual(ret0.get_counts(), {"001 00": 100})

        qc1 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc1.x(0)
        qc1.measure(0, creg[0])
        qc1.append(op, [qubit0, qubit1, qubit2], creg)
        qc1.measure_all()

        ret1 = backend.run(qc1, shots=100).result()
        self.assertSuccess(ret1)
        self.assertEqual(ret1.get_counts(), {"011 01": 100})

        qc2 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc2.x(1)
        qc2.measure(0, creg[0])
        qc2.measure(1, creg[1])
        qc2.append(op, [qubit0, qubit1, qubit2], creg)
        qc2.measure_all()

        ret2 = backend.run(qc2, shots=100).result()
        self.assertSuccess(ret2)
        self.assertEqual(ret2.get_counts(), {"110 10": 100})

        qc3 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc3.x(0)
        qc3.x(1)
        qc3.measure(0, creg[0])
        qc3.measure(1, creg[1])
        qc3.append(op, [qubit0, qubit1, qubit2], creg)
        qc3.measure_all()

        ret3 = backend.run(qc3, shots=100).result()
        self.assertSuccess(ret3)
        self.assertEqual(ret3.get_counts(), {"011 11": 100})

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_switch_with_default(self, method):
        """Test that a switch statement can be constructed with a default case at the end."""

        backend = self.backend(method=method, seed_simulator=1)

        qubit0 = Qubit()
        qubit1 = Qubit()
        qubit2 = Qubit()
        creg = ClassicalRegister(2)
        case1 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        case2.x(1)
        case3 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        case3.x(2)

        op = SwitchCaseOp(creg, [(0, case1), (1, case2), (CASE_DEFAULT, case3)])

        qc0 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc0.measure(0, creg[0])
        qc0.append(op, [qubit0, qubit1, qubit2], creg)
        qc0.measure_all()

        ret0 = backend.run(qc0, shots=100).result()
        self.assertSuccess(ret0)
        self.assertEqual(ret0.get_counts(), {"001 00": 100})

        qc1 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc1.x(0)
        qc1.measure(0, creg[0])
        qc1.append(op, [qubit0, qubit1, qubit2], creg)
        qc1.measure_all()

        ret1 = backend.run(qc1, shots=100).result()
        self.assertSuccess(ret1)
        self.assertEqual(ret1.get_counts(), {"011 01": 100})

        qc2 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc2.x(1)
        qc2.measure(0, creg[0])
        qc2.measure(1, creg[1])
        qc2.append(op, [qubit0, qubit1, qubit2], creg)
        qc2.measure_all()

        ret2 = backend.run(qc2, shots=100).result()
        self.assertSuccess(ret2)
        self.assertEqual(ret2.get_counts(), {"110 10": 100})

        qc3 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc3.x(0)
        qc3.x(1)
        qc3.measure(0, creg[0])
        qc3.measure(1, creg[1])
        qc3.append(op, [qubit0, qubit1, qubit2], creg)
        qc3.measure_all()

        ret3 = backend.run(qc3, shots=100).result()
        self.assertSuccess(ret3)
        self.assertEqual(ret3.get_counts(), {"111 11": 100})

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_switch_multiple_cases_to_same_block(self, method):
        """Test that it is possible to add multiple cases that apply to the same block, if they are
        given as a compound value.  This is an allowed special case of block fall-through."""

        backend = self.backend(method=method, seed_simulator=1)

        qubit0 = Qubit()
        qubit1 = Qubit()
        qubit2 = Qubit()
        creg = ClassicalRegister(2)
        case1 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        case2.x(1)

        creg = ClassicalRegister(2)

        op = SwitchCaseOp(creg, [(0, case1), ((1, 2), case2)])

        qc0 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc0.measure(0, creg[0])
        qc0.append(op, [qubit0, qubit1, qubit2], creg)
        qc0.measure_all()

        ret0 = backend.run(qc0, shots=100).result()
        self.assertSuccess(ret0)
        self.assertEqual(ret0.get_counts(), {"001 00": 100})

        qc1 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc1.x(0)
        qc1.measure(0, creg[0])
        qc1.append(op, [qubit0, qubit1, qubit2], creg)
        qc1.measure_all()

        ret1 = backend.run(qc1, shots=100).result()
        self.assertSuccess(ret1)
        self.assertEqual(ret1.get_counts(), {"011 01": 100})

        qc2 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc2.x(1)
        qc2.measure(0, creg[0])
        qc2.measure(1, creg[1])
        qc2.append(op, [qubit0, qubit1, qubit2], creg)
        qc2.measure_all()

        ret2 = backend.run(qc2, shots=100).result()
        self.assertSuccess(ret2)
        self.assertEqual(ret2.get_counts(), {"000 10": 100})

        qc3 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc3.x(0)
        qc3.x(1)
        qc3.measure(0, creg[0])
        qc3.measure(1, creg[1])
        qc3.append(op, [qubit0, qubit1, qubit2], creg)
        qc3.measure_all()

        ret3 = backend.run(qc3, shots=100).result()
        self.assertSuccess(ret3)
        self.assertEqual(ret3.get_counts(), {"011 11": 100})

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_switch_transpilation(self, method):
        """Test swtich test cases can be transpiled"""

        backend = self.backend(method=method, seed_simulator=1)

        qubit0 = Qubit()
        qubit1 = Qubit()
        qubit2 = Qubit()

        creg = ClassicalRegister(2)
        qc = QuantumCircuit([qubit0, qubit1, qubit2], creg)

        with qc.switch(creg) as case:
            with case(0):
                qc.x(0)
            with case(1):
                qc.x(1)
            with case(case.DEFAULT):
                qc.x(2)

        qc.measure_all()

        transpiled = transpile(qc, backend)

        ret0 = backend.run(transpiled, shots=100).result()
        self.assertSuccess(ret0)
        self.assertEqual(ret0.get_counts(), {"001 00": 100})

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_switch_register_with_classical_expression(self, method):
        """Test that a switch statement can be constructed with a register as a condition."""

        backend = self.backend(method=method, seed_simulator=1)

        qubit0 = Qubit()
        qubit1 = Qubit()
        qubit2 = Qubit()
        creg = ClassicalRegister(2)
        case1 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        case2.x(1)
        case3 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        case3.x(2)

        op = SwitchCaseOp(expr.lift(creg), [(0, case1), (1, case2), (2, case3)])

        qc0 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc0.measure(0, creg[0])
        qc0.append(op, [qubit0, qubit1, qubit2], creg)
        qc0.measure_all()

        ret0 = backend.run(qc0, shots=100).result()
        self.assertSuccess(ret0)
        self.assertEqual(ret0.get_counts(), {"001 00": 100})

        qc1 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc1.x(0)
        qc1.measure(0, creg[0])
        qc1.append(op, [qubit0, qubit1, qubit2], creg)
        qc1.measure_all()

        ret1 = backend.run(qc1, shots=1).result()
        self.assertSuccess(ret1)
        self.assertEqual(ret1.get_counts(), {"011 01": 1})

        qc2 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc2.x(1)
        qc2.measure(0, creg[0])
        qc2.measure(1, creg[1])
        qc2.append(op, [qubit0, qubit1, qubit2], creg)
        qc2.measure_all()

        ret2 = backend.run(qc2, shots=100).result()
        self.assertSuccess(ret2)
        self.assertEqual(ret2.get_counts(), {"110 10": 100})

        qc3 = QuantumCircuit([qubit0, qubit1, qubit2], creg)
        qc3.x(0)
        qc3.x(1)
        qc3.measure(0, creg[0])
        qc3.measure(1, creg[1])
        qc3.append(op, [qubit0, qubit1, qubit2], creg)
        qc3.measure_all()

        ret3 = backend.run(qc3, shots=100).result()
        self.assertSuccess(ret3)
        self.assertEqual(ret3.get_counts(), {"011 11": 100})

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_if_expr_true_body_builder(self, method):
        """test expression with branch operation"""
        backend = self.backend(method=method)

        # case creg==1
        qreg = QuantumRegister(4)
        creg = ClassicalRegister(3, "test")
        circ = QuantumCircuit(qreg, creg)
        circ.y(0)
        circ.h(circ.qubits[1:4])
        circ.barrier()
        circ.measure(0, 0)  # 001

        with circ.if_test(expr.equal(ClassicalRegister(3, "test"), 1)):
            circ.h(circ.qubits[1:4])

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("0001 001", counts)

        # case creg==3
        qreg = QuantumRegister(4)
        creg = ClassicalRegister(3, "test")
        circ = QuantumCircuit(qreg, creg)
        circ.y(0)
        circ.h(circ.qubits[1:4])
        circ.barrier()
        circ.measure(0, 0)
        circ.measure(0, 1)  # 011

        with circ.if_test(expr.equal(ClassicalRegister(3, "test"), 3)):
            circ.h(circ.qubits[1:4])

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("0001 011", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_if_expr_false_body_builder(self, method):
        """test expression with branch operation"""
        backend = self.backend(method=method)

        # case creg==1
        qreg = QuantumRegister(4)
        creg = ClassicalRegister(3, "test")
        circ = QuantumCircuit(qreg, creg)
        circ.y(0)
        circ.h(circ.qubits[1:4])
        circ.barrier()
        circ.measure(0, 0)  # 001

        with circ.if_test(expr.equal(ClassicalRegister(3, "test"), 2)) as else_:
            circ.y(0)
        with else_:
            circ.h(circ.qubits[1:4])

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("0001 001", counts)

        # case creg==3
        qreg = QuantumRegister(4)
        creg = ClassicalRegister(3, "test")
        circ = QuantumCircuit(qreg, creg)
        circ.y(0)
        circ.h(circ.qubits[1:4])
        circ.barrier()
        circ.measure(0, 0)
        circ.measure(0, 1)  # 011

        with circ.if_test(expr.equal(ClassicalRegister(3, "test"), 1)) as else_:
            circ.y(0)
        with else_:
            circ.h(circ.qubits[1:4])

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("0001 011", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_while_expr_loop_break(self, method):
        backend = self.backend(method=method)

        qreg = QuantumRegister(1)
        creg = ClassicalRegister(1)
        circ = QuantumCircuit(qreg, creg)
        circ.y(0)
        circ.measure(0, 0)

        circ_while = QuantumCircuit(qreg, creg)
        circ_while.y(0)
        circ_while.measure(0, 0)
        circ_while.break_loop()
        circ.while_loop(expr.Value(True, types.Bool()), circ_while, [0], [0])

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("0 0", counts)

        qreg = QuantumRegister(1)
        creg = ClassicalRegister(1)
        circ = QuantumCircuit(qreg, creg)
        circ.y(0)
        circ.measure(0, 0)

        circ_while = QuantumCircuit(qreg, creg)
        circ_while.y(0)
        circ_while.measure(0, 0)
        circ_while.break_loop()
        circ.while_loop(expr.Value(False, types.Bool()), circ_while, [0], [0])

        circ.measure_all()

        result = backend.run(circ, method=method).result()
        self.assertSuccess(result)

        counts = result.get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("1 1", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_bit_and_operation(self, method):
        """test bit-and operation"""
        qr = QuantumRegister(7)
        cr = ClassicalRegister(7)
        qc = QuantumCircuit(qr, cr)
        qc.x(0)
        qc.x(2)
        qc.measure(range(4), range(4))  # 0101
        qc.barrier()
        b01 = expr.bit_and(cr[0], cr[1])  # 1 & 0 -> 0
        with qc.if_test(b01):
            qc.x(4)  # q4 -> 0

        b02 = expr.bit_and(cr[0], cr[2])  # 1 & 1 -> 1
        with qc.if_test(b02):
            qc.x(5)  # q5 -> 0

        b13 = expr.bit_and(cr[1], cr[3])  # 0 & 0 -> 0
        with qc.if_test(b13):
            qc.x(6)  # q6 -> 0

        qc.measure(range(7), range(7))  # 0100101

        backend = self.backend(method=method)
        counts = backend.run(qc).result().get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("0100101", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_bit_or_operation(self, method):
        """test bit-or operation"""
        qr = QuantumRegister(7)
        cr = ClassicalRegister(7)
        qc = QuantumCircuit(qr, cr)
        qc.x(0)
        qc.x(2)
        qc.measure(range(4), range(4))  # 0101
        qc.barrier()
        b01 = expr.bit_or(cr[0], cr[1])  # 1 & 0 -> 1
        with qc.if_test(b01):
            qc.x(4)  # q4 -> 1

        b02 = expr.bit_or(cr[0], cr[2])  # 1 & 1 -> 1
        with qc.if_test(b02):
            qc.x(5)  # q5 -> 0

        b13 = expr.bit_or(cr[1], cr[3])  # 0 & 0 -> 0
        with qc.if_test(b13):
            qc.x(6)  # q6 -> 0

        qc.measure(range(7), range(7))  # 0110101

        backend = self.backend(method=method)
        counts = backend.run(qc).result().get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("0110101", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_bit_xor_operation(self, method):
        """test bit-or operation"""
        qr = QuantumRegister(8)
        cr = ClassicalRegister(8)
        qc = QuantumCircuit(qr, cr)
        qc.x(0)
        qc.x(2)
        qc.measure(range(4), range(4))  # 0101
        qc.barrier()
        b01 = expr.bit_xor(cr[0], cr[1])  # (bool) 1 & (bool) 0 -> (bool) 1
        with qc.if_test(b01):
            qc.x(4)  # q4 -> 1

        b02 = expr.bit_xor(cr[0], cr[2])  # (bool) 1 & (bool) 1 -> (bool) 0
        with qc.if_test(b02):
            qc.x(5)  # q5 -> 0

        b03 = expr.bit_xor(cr[1], cr[3])  # (bool) 0 & (bool) 0 -> (bool) 0
        with qc.if_test(b03):
            qc.x(6)  # q6 -> 0

        b04 = expr.bit_xor(
            expr.Value(True, types.Bool()), expr.Value(False, types.Bool())
        )  # (bool) 1 & (bool) 0 -> (bool) 1
        with qc.if_test(b04):
            qc.x(7)  # q7 -> 1

        qc.measure(range(8), range(8))  # 10010101

        backend = self.backend(method=method)
        counts = backend.run(qc).result().get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("10010101", counts)

        qr = QuantumRegister(7)
        cr = ClassicalRegister(7)
        qc = QuantumCircuit(qr, cr)
        qc.x(0)
        qc.x(2)
        qc.measure(range(4), range(4))  # 0101
        qc.barrier()
        try:
            b04 = expr.bit_xor(
                expr.Var(cr, types.Uint(cr.size)), expr.Var(cr, types.Uint(cr.size))
            )  # (bool) 1 ^ (uint) 0101 -> error
            self.fail("do not reach here")
        except Exception:
            pass

        qr = QuantumRegister(7)
        cr = ClassicalRegister(7)
        cr0 = ClassicalRegister(7)
        qc = QuantumCircuit(qr, cr, cr0)
        qc.x(0)
        qc.x(1)
        qc.x(2)
        qc.x(3)
        qc.measure(range(4), range(4))  # 1111
        qc.barrier()
        b05 = expr.bit_xor(
            expr.Var(cr, types.Uint(cr.size)), expr.Var(cr, types.Uint(cr.size))
        )  # (uint) 1111 ^ (uint) 1111 -> (uint) 0000
        with qc.if_test(expr.equal(b05, 0b0000000)):
            qc.x(4)  # q4 -> 1
        b06 = expr.bit_xor(
            expr.Var(cr0, types.Uint(cr0.size)), expr.Var(cr, types.Uint(cr.size))
        )  # (uint) 0000 ^ (uint) 0101 -> (uint) 1111
        with qc.if_test(expr.equal(b06, 0b0001111)):
            qc.x(5)  # q5 -> 1

        qc.measure(range(7), range(7))  # 111111

        backend = self.backend(method=method)
        counts = backend.run(qc).result().get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("0000000 0111111", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_bit_not_operation(self, method):
        """test bit-not operation"""
        qr = QuantumRegister(7)
        cr = ClassicalRegister(7)
        qc = QuantumCircuit(qr, cr)
        qc.x(0)
        qc.x(2)
        qc.measure(range(4), range(4))  # 0101
        qc.barrier()
        b01 = expr.bit_not(cr[0])  # !1 -> 0
        with qc.if_test(b01):
            qc.x(4)  # q4 -> 0

        b02 = expr.bit_not(cr[1])  # !0 -> 1
        with qc.if_test(b02):
            qc.x(5)  # q5 -> 1

        qc.measure(range(7), range(7))  # 0100101

        backend = self.backend(method=method)
        counts = backend.run(qc).result().get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("0100101", counts)

        qr = QuantumRegister(7)
        cr = ClassicalRegister(7)
        qc = QuantumCircuit(qr, cr)
        qc.x(0)
        qc.x(2)
        qc.measure(range(4), range(4))  # 0101
        qc.barrier()
        b01 = expr.bit_not(expr.Var(cr, types.Uint(cr.size)))  # 0b0000101 -> 0b1111010
        with qc.if_test(expr.equal(b01, 0b1111010)):
            qc.x(4)  # q4 -> 1

        qc.measure(range(7), range(7))  # 0010101

        backend = self.backend(method=method)
        counts = backend.run(qc).result().get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("0010101", counts)

    @data("statevector", "density_matrix", "matrix_product_state", "stabilizer")
    def test_store_simple(self, method):
        """test store operation"""
        backend = self.backend(method=method)

        # Check stored values can be sampled
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        qc = QuantumCircuit(qr, cr)
        qc.x(2)
        qc.measure(range(4), range(4))
        qc.store(cr, 0b1000)  # measured classical registers are modified

        counts = backend.run(qc).result().get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("1000", counts)

        # Check stored values to creg can be evaluated
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        qc = QuantumCircuit(qr, cr)
        qc.x(2)
        qc.measure(range(4), range(4))
        qc.store(cr, 0b1000)  # override
        with qc.if_test((2, False)):
            # must reach
            qc.x(1)  # 0b1010
        qc.measure(range(4), range(4))

        counts = backend.run(qc).result().get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("0110", counts)

        # Check stored values to clbit can be evaluated
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        qc = QuantumCircuit(qr, cr)
        qc.x(2)
        qc.measure(range(4), range(4))
        qc.store(cr[2], False)  # override
        with qc.if_test((2, False)):
            # must reach
            qc.x(1)  # 0b1010
        qc.measure(range(4), range(4))

        counts = backend.run(qc).result().get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("0110", counts)

        # Check stored values can be stored
        qr = QuantumRegister(4)
        cr0 = ClassicalRegister(4)
        cr1 = ClassicalRegister(4)
        qc = QuantumCircuit(qr, cr0, cr1)
        qc.x(2)
        qc.measure(range(4), range(4))
        qc.store(cr0, 0b1000)  # measured classical registers are modified
        qc.store(cr1, cr0)  # measured classical registers are modified

        counts = backend.run(qc).result().get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("1000 1000", counts)

    def test_bit_mapping_in_compiler(self):
        """Test different bit mappings are correctly inlined"""
        parent = QuantumCircuit(5, 2)
        parent.x(0)
        parent.measure(0, 0)

        true_body = QuantumCircuit(1, 0)
        true_body.x(0)

        parent.append(IfElseOp((parent.clbits[0], 1), true_body), [1], [])

        parent.measure(1, 1)

        simulator = self.backend()
        counts = simulator.run(parent).result().get_counts()
        self.assertEqual(len(counts), 1)
        self.assertIn("11", counts)
