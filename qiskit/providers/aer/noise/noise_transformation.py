# coding: utf-8

import numpy
import sympy
import itertools

from qiskit.providers.aer.noise.errors import QuantumError
from qiskit.providers.aer.noise.errors.errorutils import standard_gate_unitary
from qiskit.quantum_info.operators.channel import Kraus

# #only for 1-qubit errors for now
# def quantum_error_to_kraus_operators(error):
#     qubits_num = 1
#     error_ops = []
#     for noise_circuit, noise_prob in zip(error._noise_circuits, error._noise_probabilities):
#         noise_circuit_ops = [numpy.eye(2 ** qubits_num)]
#         for noise_circuit_element in noise_circuit:
#             std_op = standard_gate_unitary(noise_circuit_element['name'])
#             if std_op is not None:
#                 kraus_matrices = [std_op]
#             if noise_circuit_element['name'] == 'kraus':
#                 kraus_matrices = noise_circuit_element['params']
#             if noise_circuit_element['name'] == 'unitary':
#                 #TODO: I expect only one matrix here. Is this assumption correct?
#                 kraus_matrices = [noise_circuit_element['params']]
#             if kraus_matrices is None:
#                 raise "Could not understand the error {}".format(error)
#             kraus_matrices = [numpy.array(matrix) for matrix in kraus_matrices]
#             noise_circuit_ops = [b @ a for (a, b) in itertools.product(noise_circuit_ops, kraus_matrices)]
#         error_ops += [numpy.sqrt(noise_prob) * noise_op for noise_op in noise_circuit_ops]
#     return error_ops

def approximate_quantum_error(error,
                              operator_string = None,
                              operator_dict = None,
                              operator_list = None):
    """Return an approximate QuantumError.

    Args:
        error (QuantumError): the error to be approximated.

    Returns:
        QuantumError: the approximate quantum error.
    """
    # This would accept as input anything that the QuantumError class can take as input
    # Eg: list of circuits, list of kraus matrices, a QuantumChannel sublcass object, or a
    # QuantumError object. This could be done as:
    if not isinstance(error, QuantumError):
        error = QuantumError(error)
    error_kraus_operators = Kraus(error.to_channel()).data
    transformer = NoiseTransformer()
    if operator_string is not None:
        operator_dict = transformer.named_operators[operator_string]
    if operator_dict is not None:
        names, operator_list = zip(*operator_dict.items())
    if operator_list is not None:
        probabilities = transformer.transform_by_operator_list(operator_list, error_kraus_operators)
        identity_prob = 1 - sum(probabilities)
        if identity_prob < 0 or identity_prob > 1:
            raise RuntimeError("Approximated channel operators probabilities sum to {}".format(1-identity_prob))
        quantum_error_spec = [([{'name': 'id', 'qubits':[0]}],identity_prob)]
        for (op_matrices, probability) in zip(operator_list, probabilities):
            quantum_error_spec.append(([{'name': 'kraus',
                                        'qubits': [0],
                                        'params': op_matrices}
                                       ], probability))
        return QuantumError(quantum_error_spec)

    raise Exception("Quantum error approximation failed - no approximating operators detected")


def approximate_noise_model(model, **kwargs):
    """Return an approximate noise model.

    Args:
        model (NoiseModel): the noise model to be approximated.

    Returns:
        NoiseModel: the approximate noise model.

    Raises:
        NoiseError: if the QuantumError cannot be approximated.
    """
    # This function would iterate over all QuantumErrors in the noise model
    # and call the `approximate_quantum_errror` function on them to generate
    # an approximate noise model
    # TODO: Raise error about 2-qubit errors
    approx_noise_model = None
    return approx_noise_model

class NoiseTransformer:
    """Transforms one quantum noise channel to another based on a specified criteria.

    A quantum 1-qubit noise channel is represented by Kraus operators: a sequence (E0, E1,...,En) of
    2x2 matrices that satisfy \sum_{i=0}^n E_i^\daggerE_i = I

    Given a quantum state's density function rho, the effect of the channel on this state is
    rho -> \sum_{i=1}^n E_i * rho * E_i^\dagger

    The goal of this module is to transform one noise channel into another, where the goal channel
    is constructed from a given set of matrices, using coefficients computed by the module in order
    to satisfy some criteria (for now, we wish the output channel to be "close" to the input channel)

    The main public function of the module is transform(), and a conversion of qobj inputs is given
    via the transform_qobj() function.
    """

    def __init__(self):
        named_operators = {
            'pauli': {'X': standard_gate_unitary('x'),
                      'Y': standard_gate_unitary('y'),
                      'Z': standard_gate_unitary('z')
                      },
            'reset': {
                'p': (numpy.array([[1, 0], [0, 0]]), numpy.array([[0, 1], [0, 0]])),
                'q': (numpy.array([[0, 0], [0, 1]]), numpy.array([[0, 0], [1, 0]])),
            },
            # 'clifford': self.single_qubit_full_clifford_group()
        }

        self.fidelity_data = None
        self.use_honesty_constraint = True
        self.noise_kraus_operators = None
        self.transform_channel_operators = None

    # transformation interface methods

    def transform_by_operator_list(self, transform_channel_operators, noise_kraus_operators):
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
        full_transform_channel_operators = self.prepare_channel_operator_list(self.transform_channel_operators)
        channel_matrices, const_channel_matrix = self.generate_channel_matrices(full_transform_channel_operators)
        self.prepare_honesty_constraint(full_transform_channel_operators)
        probabilities = self.transform_by_given_channel(channel_matrices, const_channel_matrix)
        return probabilities

    # convenience wrapper method
    def transform_by_operator_dictionary(self, transform_channel_operators_dictionary, noise_kraus_operators):
        names, operators = zip(*transform_channel_operators_dictionary.items())
        probabilities = self.transform_by_operator_list(operators, noise_kraus_operators)
        return dict(zip(names, probabilities))

    # convert to sympy matrices and verify that each singleton is in a tuple; also add identity matrix
    @staticmethod
    def prepare_channel_operator_list(ops_list):
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
        coefficients = [self.fidelity(ops) for ops in transform_channel_operators_list]
        self.fidelity_data = {
            'goal': goal,
            'coefficients': coefficients[1:]  # coefficients[0] corresponds to I
        }

    # methods relevant to the transformation to quadratic programming instance

    @staticmethod
    def fidelity(channel): #TODO replace with qiskit fidelity
        return sum([numpy.abs(numpy.trace(E)) ** 2 for E in channel])

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

        symbols_string = " ".join(["x{}".format(i) for i in range(len(transform_channel_operators_list))])
        symbols = sympy.symbols(symbols_string, real=True, positive=True)
        exp = symbols[1] # exp will contain the symbolic expression "x1 +...+ xn"
        for i in range(2, len(symbols)):
            exp = symbols[i] + exp
        # symbolic_operators_list is a list of lists; we flatten it the next line
        symbolic_operators_list = [[sympy.sqrt(symbols[i]) * op for op in ops]
                                   for (i, ops) in enumerate(transform_channel_operators_list)]
        symbolic_operators = [op for ops in symbolic_operators_list for op in ops]
        # channel_matrix_representation() peforms the required linear algebra to find the representing matrices.
        operators_channel = self.channel_matrix_representation(symbolic_operators).subs(symbols[0], 1 - exp)
        return self.generate_channel_quadratic_programming_matrices(operators_channel, symbols[1:])

    @staticmethod
    def compute_channel_operation(rho, operators):
        """
        Given a quantum state's density function rho, the effect of the channel on this state is
        rho -> \sum_{i=1}^n E_i * rho * E_i^\dagger
        """

        return sum([E * rho * E.H for E in operators], sympy.zeros(operators[0].rows))

    @staticmethod
    def flatten_matrix(m):
        return [element for element in m]

    def channel_matrix_representation(self, operators):
        """
        We convert the operators to a matrix by applying the channel to the four basis elements of
        the 2x2 matrix space representing density operators; this is standard linear algebra
        """
        standard_base = [
            sympy.Matrix([[1, 0], [0, 0]]),
            sympy.Matrix([[0, 1], [0, 0]]),
            sympy.Matrix([[0, 0], [1, 0]]),
            sympy.Matrix([[0, 0], [0, 1]])
        ]
        return (sympy.Matrix([self.flatten_matrix(self.compute_channel_operation(rho, operators))
                              for rho in standard_base]))

    def generate_channel_quadratic_programming_matrices(self, channel, symbols):
        """
        Args:
             channel: a 4x4 symbolic matrix
             symbols: the symbols x1, ..., xn which may occur in the matrix

        Output:
            A list of 4x4 complex matrices ([D1, D2, ..., Dn], E) such that:
            channel == x1*D1 + ... + xn*Dn + E
        """
        return (
            [self.get_matrix_from_channel(channel, symbol) for symbol in symbols],
            self.get_const_matrix_from_channel(channel, symbols)
        )

    @staticmethod
    def get_matrix_from_channel(channel, symbol):
        """
        Args:
            channel: a 4x4 symbolic matrix.
            Each entry is assumed to be a polynomial of the form a1x1 + ... + anxn + c
            symbol: a symbol xi

        Output:
            A 4x4 numerical matrix where for each entry,
            if a1x1 + ... + anxn + c was the corresponding entry in the input channel
            then the corresponding entry in the output matrix is ai.
        """
        n = channel.rows
        M = numpy.zeros((n, n), dtype=numpy.complex_)
        for (i, j) in itertools.product(range(n), range(n)):
            M[i, j] = numpy.complex(sympy.Poly(channel[i, j], symbol).coeff_monomial(symbol))
        return M

    @staticmethod
    def get_const_matrix_from_channel(channel, symbols):
        """
                Args:
                    channel: a 4x4 symbolic matrix.
                    Each entry is assumed to be a polynomial of the form a1x1 + ... + anxn + c
                    symbols: The full list [x1, ..., xn] of symbols used in the matrix

                Output:
                    A 4x4 numerical matrix where for each entry,
                    if a1x1 + ... + anxn + c was the corresponding entry in the input channel
                    then the corresponding entry in the output matrix is c.
                """
        n = channel.rows
        M = numpy.zeros((n, n), dtype=numpy.complex_)
        for (i, j) in itertools.product(range(n), range(n)):
            M[i, j] = numpy.complex(sympy.Poly(channel[i, j], symbols).coeff_monomial(1))
        return M

    def transform_by_given_channel(self, channel_matrices, const_channel_matrix):
        """
        This method creates the quadratic programming instance for minimizing the Hilbert-Schmidt
        norm of the matrix (A-B) obtained as the difference of the input noise channel and the
        output channel we wish to determine.
        """
        target_channel_matrix = self.numeric_channel_matrix_representation(self.noise_kraus_operators)
        const_matrix = const_channel_matrix - target_channel_matrix
        P = self.compute_P(channel_matrices)
        q = self.compute_q(channel_matrices, const_matrix)
        return self.solve_quadratic_program(P, q)

    #TODO: replace with Kraus object from Terra ("evolve" method)
    # qiskit-terra\qiskit\quantum_info\operators\channel
    @staticmethod
    def numeric_compute_channel_operation(rho, operators):
        return numpy.sum([E.dot(rho).dot(E.conj().T) for E in operators], 0)

    def numeric_channel_matrix_representation(self, operators):
        standard_base = [
            numpy.array([[1, 0], [0, 0]]),
            numpy.array([[0, 1], [0, 0]]),
            numpy.array([[0, 0], [1, 0]]),
            numpy.array([[0, 0], [0, 1]])
        ]
        return (numpy.array([self.numeric_compute_channel_operation(rho, operators).flatten()
                              for rho in standard_base]))

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
            #TODO: consider using QP with CVXPY
            import cvxopt
        except ImportError:
            raise ImportError("The CVXOPT library is required to use this module")
        P = cvxopt.matrix(numpy.array(P).astype(float))
        q = cvxopt.matrix(numpy.array(q).astype(float)).T
        n = len(q)
        # G and h constrain:
        #   1) sum of probs is less then 1
        #   2) All probs bigger than 0
        #   3) Honesty (measured using fidelity, if given)
        G_data = [[1] * n] + [([-1 if i == k else 0 for i in range(n)]) for k in range(n)]
        h_data = [1] + [0] * n
        if self.fidelity_data is not None:
            G_data.append(self.fidelity_data['coefficients'])
            h_data.append(self.fidelity_data['goal'])
        G = cvxopt.matrix(numpy.array(G_data).astype(float))
        h = cvxopt.matrix(numpy.array(h_data).astype(float))
        cvxopt.solvers.options['show_progress'] = False
        return cvxopt.solvers.qp(P, q, G, h)['x']