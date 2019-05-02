# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Noise transformation module

The goal of this module is to transform one 1-qubit noise channel
(given by the QuantumError class) into another, built from specified
"building blocks" (given as Kraus matrices) such that the new channel is
as close as possible to the original one in the Hilber-Schmidt metric.

For a typical use case, consider a simulator for circuits built from the
Clifford group. Computations on such circuits can be simulated at
polynomial time and space, but not all noise channels can be used in such
a simulation. To enable noisy Clifford simulation one can transform the
given noise channel into the closest one, Hilbert-Schmidt wise, that can be
used in a Clifford simulator.
"""

import itertools
import numpy
import sympy

from qiskit.providers.aer.noise.errors import QuantumError
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.noiseerror import NoiseError
from qiskit.providers.aer.noise.errors.errorutils import single_qubit_clifford_instructions
from qiskit.quantum_info.operators.channel import Kraus
from qiskit.quantum_info.operators.channel import SuperOp


def approximate_quantum_error(error, *,
                              operator_string=None,
                              operator_dict=None,
                              operator_list=None):
    """Return an approximate QuantumError bases on the Hilbert-Schmidt metric.

    Currently this is only implemented for 1-qubit QuantumErrors.

    Args:
        error (QuantumError): the error to be approximated.
        operator_string (string or None): a name for a premade set of
            building blocks for the output channel (Default: None).
        operator_dict (dict or None): a dictionary whose values are the
            building blocks for the output channel (Default: None).
        operator_list (dict or None): list of building blocks for the
            output channel (Default: None).

    Returns:
        QuantumError: the approximate quantum error.

    Raises:
        NoiseError: if number of qubits is not supported or approximation
        failsed.

    Additional Information
    ----------------------
    The operator input precedence is as follows: list < dict < string
    if a string is given, dict is overwritten; if a dict is given, list is
    overwritten possible values for string are 'pauli', 'reset', 'clifford'
    For further information see `NoiseTransformer.named_operators`.
    """

    if not isinstance(error, QuantumError):
        error = QuantumError(error)
    if error.number_of_qubits > 1:
        raise NoiseError("Only 1-qubit noises can be converted, {}-qubit "
                         "noise found in model".format(error.number_of_qubits))

    error_kraus_operators = Kraus(error.to_quantumchannel()).data
    transformer = NoiseTransformer()
    if operator_string is not None:
        operator_string = operator_string.lower()
        if operator_string not in transformer.named_operators.keys():
            raise RuntimeError(
                "No information about noise type {}".format(operator_string))
        operator_dict = transformer.named_operators[operator_string]
    if operator_dict is not None:
        names, operator_list = zip(*operator_dict.items())
    if operator_list is not None:
        op_matrix_list = [
            transformer.operator_matrix(operator) for operator in operator_list
        ]
        probabilities = transformer.transform_by_operator_list(
            op_matrix_list, error_kraus_operators)
        identity_prob = 1 - sum(probabilities)
        if identity_prob < 0 or identity_prob > 1:
            raise RuntimeError(
                "Approximated channel operators probabilities sum to {}".
                format(1 - identity_prob))
        quantum_error_spec = [([{'name': 'id', 'qubits': [0]}], identity_prob)]
        op_circuit_list = [
            transformer.operator_circuit(operator)
            for operator in operator_list
        ]
        for (operator, probability) in zip(op_circuit_list, probabilities):
            quantum_error_spec.append((operator, probability))
        return QuantumError(quantum_error_spec)

    raise NoiseError(
        "Quantum error approximation failed - no approximating operators detected"
    )


def approximate_noise_model(model, *,
                            operator_string=None,
                            operator_dict=None,
                            operator_list=None):
    """Return an approximate noise model.

    Args:
        model (NoiseModel): the noise model to be approximated.
        operator_string (string or None): a name for a premade set of
            building blocks for the output channel (Default: None).
        operator_dict (dict or None): a dictionary whose values are the
            building blocks for the output channel (Default: None).
        operator_list (dict or None): list of building blocks for the
            output channel (Default: None).

    Returns:
        NoiseModel: the approximate noise model.

    Raises:
        NoiseError: if number of qubits is not supported or approximation
        failsed.

    Additional Information
    ----------------------
    The operator input precedence is as follows: list < dict < string
    if a string is given, dict is overwritten; if a dict is given, list is
    overwritten possible values for string are 'pauli', 'reset', 'clifford'
    For further information see `NoiseTransformer.named_operators`.
    """

    #We need to iterate over all the errors in the noise model.
    #No nice interface for this now, easiest way is to mimic as_dict

    error_list = []
    # Add default quantum errors
    for operation, error in model._default_quantum_errors.items():
        error = approximate_quantum_error(
            error,
            operator_string=operator_string,
            operator_dict=operator_dict,
            operator_list=operator_list)
        error_dict = error.as_dict()
        error_dict["operations"] = [operation]
        error_list.append(error_dict)

    # Add specific qubit errors
    for operation, qubit_dict in model._local_quantum_errors.items():
        for qubits_str, error in qubit_dict.items():
            error = approximate_quantum_error(
                error,
                operator_string=operator_string,
                operator_dict=operator_dict,
                operator_list=operator_list)
            error_dict = error.as_dict()
            error_dict["operations"] = [operation]
            error_dict["gate_qubits"] = [model._str2qubits(qubits_str)]
            error_list.append(error_dict)

    # Add non-local errors
    for operation, qubit_dict in model._nonlocal_quantum_errors.items():
        for qubits_str, noise_dict in qubit_dict.items():
            for noise_str, error in noise_dict.items():
                error = approximate_quantum_error(
                    error,
                    operator_string=operator_string,
                    operator_dict=operator_dict,
                    operator_list=operator_list)
                error_dict = error.as_dict()
                error_dict["operations"] = [operation]
                error_dict["gate_qubits"] = [model._str2qubits(qubits_str)]
                error_dict["noise_qubits"] = [model._str2qubits(noise_str)]
                error_list.append(error_dict)

    # Add default readout error
    if model._default_readout_error is not None:
        error = approximate_quantum_error(
            model._default_readout_error,
            operator_string=operator_string,
            operator_dict=operator_dict,
            operator_list=operator_list)
        error_dict = error.as_dict()
        error_list.append(error_dict)

    # Add local readout error
    for qubits_str, error in model._local_readout_errors.items():
        error = approximate_quantum_error(
            error,
            operator_string=operator_string,
            operator_dict=operator_dict,
            operator_list=operator_list)
        error_dict = error.as_dict()
        error_dict["gate_qubits"] = [model._str2qubits(qubits_str)]
        error_list.append(error_dict)

    approx_noise_model = NoiseModel.from_dict({
        "errors": error_list,
        "x90_gates": model._x90_gates
    })
    # Update basis gates
    approx_noise_model._basis_gates = model._basis_gates
    return approx_noise_model


class NoiseTransformer:
    """Transforms one quantum channel to another based on a specified criteria."""

    def __init__(self):
        self.named_operators = {
            'pauli': {
                'X': [{'name': 'x', 'qubits': [0]}],
                'Y': [{'name': 'y', 'qubits': [0]}],
                'Z': [{'name': 'z', 'qubits': [0]}]
            },
            'reset': {
                'p': [{'name': 'reset', 'qubits': [0]}],  # reset to |0>
                'q': [{'name': 'reset', 'qubits': [0]},
                      {'name': 'x', 'qubits': [0]}]  # reset to |1>
            },
            'clifford': dict([(j, single_qubit_clifford_instructions(j))
                              for j in range(1, 24)])
        }
        self.fidelity_data = None
        self.use_honesty_constraint = True
        self.noise_kraus_operators = None
        self.transform_channel_operators = None

    def operator_matrix(self, operator):
        """Converts an operator representation to Kraus matrix representation

        Args:
            operator (operator): operator representation. Can be a noise
                circuit or a matrix or a list of matrices.

        Returns:
            Kraus: the operator, converted to Kraus representation.
        """
        if isinstance(operator, list) and isinstance(operator[0], dict):
            operator_error = QuantumError([(operator, 1)])
            kraus_rep = Kraus(operator_error.to_quantumchannel()).data
            return kraus_rep
        return operator

    def operator_circuit(self, operator):
        """Converts an operator representation to noise circuit
        Args:
            operator (operator): operator representation. Can be a noise
                circuit or a matrix or a list of matrices.
        Output:
            List: The operator, converted to noise circuit representation.
        """
        if isinstance(operator, numpy.ndarray):
            return [{'name': 'unitary', 'qubits': [0], 'params': [operator]}]

        if isinstance(operator, list) and isinstance(operator[0],
                                                     numpy.ndarray):
            if len(operator) == 1:
                return [{'name': 'unitary', 'qubits': [0], 'params': operator}]
            else:
                return [{'name': 'kraus', 'qubits': [0], 'params': operator}]

        return operator

    # transformation interface methods
    def transform_by_operator_list(self, transform_channel_operators,
                                   noise_kraus_operators):
        """
        Args:
            noise_kraus_operators: a list of matrices (Kraus operators) for the input channel
            transform_channel_operators: a list of matrices or tuples of matrices
            representing Kraus operators that can construct the output channel
            e.g. [X,Y,Z] represent the Pauli channel
            and [(|0><0|, |0><1|), |1><0|, |1><1|)] represents the relaxation channel

        Output:
            A list of amplitudes that define the output channel.
            In the case the input is a list [A1, A2, ..., An] of transform matrices
            and [E0, E1, ..., Em] of noise kraus operators, the output is
            a list [p1, p2, ..., pn] of probabilities such that:
            1) p_i >= 0
            2) p1 + ... + pn <= 1
            3) [sqrt(p1)A1, sqrt(p2)A2, ..., sqrt(pn)An, sqrt(1-(p1 + ... + pn))I] is
                a list of kraus operators that define the output channel
                (which is "close" to the input chanel given by [E0, ..., Em])

            This channel can be thought of as choosing the operator Ai in probability pi and applying
            this operator to the quantum state.

            More generally, if the input is a list of tuples (not neccesarily of the same size):
            [(A1, B1, ...), (A2, B2, ...), ... (An, Bn, ...)] then the output is
            still a list [p1, p2, ..., pn] and now the output channel is defined by the operators
            [sqrt(p1)A1, sqrt(p1)B1, ..., sqrt(pn)An, sqrt(pn)Bn, ..., sqrt(1-(p1 + ... + pn))I]
        """
        self.noise_kraus_operators = noise_kraus_operators
        self.transform_channel_operators = transform_channel_operators
        full_transform_channel_operators = self.prepare_channel_operator_list(
            self.transform_channel_operators)
        channel_matrices, const_channel_matrix = self.generate_channel_matrices(
            full_transform_channel_operators)
        self.prepare_honesty_constraint(full_transform_channel_operators)
        probabilities = self.transform_by_given_channel(
            channel_matrices, const_channel_matrix)
        return probabilities

    @staticmethod
    def prepare_channel_operator_list(ops_list):
        # convert to sympy matrices and verify that each singleton is
        # in a tuple; also add identity matrix
        result = [[sympy.eye(2)]]
        for ops in ops_list:
            if not isinstance(ops, tuple) and not isinstance(ops, list):
                ops = [ops]
            result.append([sympy.Matrix(op) for op in ops])
        return result

    def prepare_honesty_constraint(self, transform_channel_operators_list):
        if not self.use_honesty_constraint:
            return
        goal = self.fidelity(self.noise_kraus_operators)
        coefficients = [
            self.fidelity(ops) for ops in transform_channel_operators_list
        ]
        self.fidelity_data = {
            'goal': goal,
            'coefficients':
            coefficients[1:]  # coefficients[0] corresponds to I
        }

    # methods relevant to the transformation to quadratic programming instance

    @staticmethod
    def fidelity(channel):
        return sum([numpy.abs(numpy.trace(E))**2 for E in channel])

    def generate_channel_matrices(self, transform_channel_operators_list):
        """
        Generates a list of 4x4 symbolic matrices describing the channel defined from the given operators

        Args:
             transform_channel_operators_list: a list of tuples of matrices which represent Kraus operators
             The identity matrix is assumed to be the first element in the list
             [(I, ), (A1, B1, ...), (A2, B2, ...), ..., (An, Bn, ...)]
             e.g. for a Pauli channel, the matrices are
             [(I,), (X,), (Y,), (Z,)]
             for relaxation they are
             [(I, ), (|0><0|, |0><1|), |1><0|, |1><1|)]

        We consider this input to symbolically represent a channel in the following manner:
        define indeterminates x0, x1, ..., xn which are meant to represent probabilities
        such that xi >=0 and x0 = 1-(x1 + ... + xn)
        Now consider the quantum channel defined via the Kraus operators
        {sqrt(x0)I, sqrt(x1)A1, sqrt(x1)B1, ..., sqrt(xn)An, sqrt(xn)Bn, ...}
        This is the channel C symbolically represented by the operators


        Output:
            A list of 4x4 complex matrices ([D1, D2, ..., Dn], E) such that:
            The matrix x1*D1 + ... + xn*Dn + E represents the operation of the channel C on the density operator
            we find it easier to work with this representation of C when performing the combinatorial optimization
        """

        symbols_string = " ".join([
            "x{}".format(i)
            for i in range(len(transform_channel_operators_list))
        ])
        symbols = sympy.symbols(symbols_string, real=True, positive=True)
        exp = symbols[
            1]  # exp will contain the symbolic expression "x1 +...+ xn"
        for i in range(2, len(symbols)):
            exp = symbols[i] + exp
        # symbolic_operators_list is a list of lists; we flatten it the next line
        symbolic_operators_list = [[
            sympy.sqrt(symbols[i]) * op for op in ops
        ] for (i, ops) in enumerate(transform_channel_operators_list)]
        symbolic_operators = [
            op for ops in symbolic_operators_list for op in ops
        ]
        # channel_matrix_representation() peforms the required linear
        # algebra to find the representing matrices.
        operators_channel = self.channel_matrix_representation(
            symbolic_operators).subs(symbols[0], 1 - exp)
        return self.generate_channel_quadratic_programming_matrices(
            operators_channel, symbols[1:])

    @staticmethod
    def compute_channel_operation(rho, operators):
        # Given a quantum state's density function rho, the effect of the
        # channel on this state is
        # rho -> \sum_{i=1}^n E_i * rho * E_i^\dagger
        return sum([E * rho * E.H for E in operators],
                   sympy.zeros(operators[0].rows))

    @staticmethod
    def flatten_matrix(m):
        return [element for element in m]

    def channel_matrix_representation(self, operators):
        # We convert the operators to a matrix by applying the channel to
        # the four basis elements of the 2x2 matrix space representing
        # density operators; this is standard linear algebra
        standard_base = [
            sympy.Matrix([[1, 0], [0, 0]]),
            sympy.Matrix([[0, 1], [0, 0]]),
            sympy.Matrix([[0, 0], [1, 0]]),
            sympy.Matrix([[0, 0], [0, 1]])
        ]
        return (sympy.Matrix([
            self.flatten_matrix(
                self.compute_channel_operation(rho, operators))
            for rho in standard_base
        ]))

    def generate_channel_quadratic_programming_matrices(
            self, channel, symbols):
        """
        Args:
             channel: a 4x4 symbolic matrix
             symbols: the symbols x1, ..., xn which may occur in the matrix

        Output:
            A list of 4x4 complex matrices ([D1, D2, ..., Dn], E) such that:
            channel == x1*D1 + ... + xn*Dn + E
        """
        return ([
            self.get_matrix_from_channel(channel, symbol) for symbol in symbols
        ], self.get_const_matrix_from_channel(channel, symbols))

    @staticmethod
    def get_matrix_from_channel(channel, symbol):
        """Extract the numeric parameter matrix.

        Args:
            channel (matrix): a 4x4 symbolic matrix.
            symbol (list): a symbol xi

        Returns
            matrix: a 4x4 numeric matrix.

        Additional Information
        ----------------------
        Each entry of the 4x4 symbolic input channel matrix is assumed to
        be a polynomial of the form a1x1 + ... + anxn + c. The corresponding
        entry in the output numeric matrix is ai.
        """
        n = channel.rows
        M = numpy.zeros((n, n), dtype=numpy.complex_)
        for (i, j) in itertools.product(range(n), range(n)):
            M[i, j] = numpy.complex(
                sympy.Poly(channel[i, j], symbol).coeff_monomial(symbol))
        return M

    @staticmethod
    def get_const_matrix_from_channel(channel, symbols):
        """Extract the numeric constant matrix.

        Args:
            channel (matrix): a 4x4 symbolic matrix.
            symbols (list): The full list [x1, ..., xn] of symbols
                used in the matrix.

        Returns
            matrix: a 4x4 numeric matrix.

        Additional Information
        ----------------------
        Each entry of the 4x4 symbolic input channel matrix is assumed to
        be a polynomial of the form a1x1 + ... + anxn + c. The corresponding
        entry in the output numeric matrix is c.
        """
        n = channel.rows
        M = numpy.zeros((n, n), dtype=numpy.complex_)
        for (i, j) in itertools.product(range(n), range(n)):
            M[i, j] = numpy.complex(
                sympy.Poly(channel[i, j], symbols).coeff_monomial(1))
        return M

    def transform_by_given_channel(self, channel_matrices,
                                   const_channel_matrix):
        # This method creates the quadratic programming instance for
        # minimizing the Hilbert-Schmidt norm of the matrix (A-B) obtained
        # as the difference of the input noise channel and the output
        # channel we wish to determine.
        target_channel = SuperOp(Kraus(self.noise_kraus_operators))
        target_channel_matrix = target_channel._data.T

        const_matrix = const_channel_matrix - target_channel_matrix
        P = self.compute_P(channel_matrices)
        q = self.compute_q(channel_matrices, const_matrix)
        return self.solve_quadratic_program(P, q)

    def compute_P(self, As):
        vs = [numpy.array(A).flatten() for A in As]
        n = len(vs)
        P = sympy.zeros(n, n)
        for (i, j) in itertools.product(range(n), range(n)):
            P[i, j] = 2 * numpy.real(numpy.dot(vs[i], numpy.conj(vs[j])))
        return P

    def compute_q(self, As, C):
        vs = [numpy.array(A).flatten() for A in As]
        vC = numpy.array(C).flatten()
        n = len(vs)
        q = sympy.zeros(1, n)
        for i in range(n):
            q[i] = 2 * numpy.real(numpy.dot(numpy.conj(vC), vs[i]))
        return q

    # the following method is the only place in the code where we rely on the cvxopt library
    # should we consider another library, only this method needs to change
    def solve_quadratic_program(self, P, q):
        try:
            import cvxopt
        except ImportError:
            raise ImportError(
                "The CVXOPT library is required to use this module")
        P = cvxopt.matrix(numpy.array(P).astype(float))
        q = cvxopt.matrix(numpy.array(q).astype(float)).T
        n = len(q)
        # G and h constrain:
        #   1) sum of probs is less then 1
        #   2) All probs bigger than 0
        #   3) Honesty (measured using fidelity, if given)
        G_data = [[1] * n] + [([-1 if i == k else 0 for i in range(n)])
                              for k in range(n)]
        h_data = [1] + [0] * n
        if self.fidelity_data is not None:
            G_data.append(self.fidelity_data['coefficients'])
            h_data.append(self.fidelity_data['goal'])
        G = cvxopt.matrix(numpy.array(G_data).astype(float))
        h = cvxopt.matrix(numpy.array(h_data).astype(float))
        cvxopt.solvers.options['show_progress'] = False
        return cvxopt.solvers.qp(P, q, G, h)['x']