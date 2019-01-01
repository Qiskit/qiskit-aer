# coding: utf-8

import numpy
import sympy
import itertools
import copy

from functools import singledispatch


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
        # the premade operators can be accessed by calling transform() with the given name string
        self.premade_operators = {
            'pauli': {'X': numpy.array([[0, 1], [1, 0]]),
                      'Y': numpy.array([[0, -1j], [1j, 0]]),
                      'Z': numpy.array([[1, 0], [0, -1]])
                      },
            'relaxation': {
                'p': (numpy.array([[1, 0], [0, 0]]), numpy.array([[0, 1], [0, 0]])),
                'q': (numpy.array([[0, 0], [0, 1]]), numpy.array([[0, 0], [1, 0]])),
            },
            'clifford': self.single_qubit_full_clifford_group()
        }
        self.transform = singledispatch(self.transform)
        self.transform.register(list, self.transform_by_operator_list)
        self.transform.register(tuple, self.transform_by_operator_list)
        self.transform.register(str, self.transform_by_operator_string)
        self.transform.register(dict, self.transform_by_operator_dictionary)
        self.fidelity_data = None
        self.use_honesty_constraint = True
        self.noise_kraus_operators = None
        self.transform_channel_operators = None

    # transformation interface methods

    def transform(self, transform_channel_operators, noise_kraus_operators):
        """Transforms the general noise given as a list of noise_kraus_operators using the operators
        given via transform_channel_operators which can be list, dict or string"""
        raise RuntimeError("{} is not an appropriate input to transform".format(transform_channel_operators))

    def transform_by_operator_list(self, transform_channel_operators, noise_kraus_operators):
        """Main transformation function; the other transformation functions use this one

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

    def transform_by_operator_string(self, transform_channel_operators_string, noise_kraus_operators):
        if transform_channel_operators_string in self.premade_operators:
            return self.transform_by_operator_dictionary(
                self.premade_operators[transform_channel_operators_string], noise_kraus_operators)
        raise RuntimeError("No information about noise type {}".format(transform_channel_operators_string))

    def transform_by_operator_dictionary(self, transform_channel_operators_dictionary, noise_kraus_operators):
        names, operators = zip(*transform_channel_operators_dictionary.items())
        probabilities = self.transform_by_operator_list(operators, noise_kraus_operators)
        return dict(zip(names, probabilities))

    def qobj_noise_to_kraus_operators(self, noise):
        # Noises are given as a list of probabilities [p1, p2, ..., pn] and circuits [C1, C2, ..., Cn]
        # Each circuit consists of a list of noise operators that are applied sqeuentially
        # If (E_0, E_1, ..., E_n) and (F_0, F_1, ...., F_m) are applied sequentially, we can "merge"
        # them into one noise with operators (E_iF_j) for 0 <= i <= n and 0 <= j <= m
        # This is then multiplied by sqrt(pk) for the circuit k
        probabilities = noise['probabilities']
        circuits = noise['instructions']
        operators_for_circuits = [self.noise_circuit_to_kraus_operators(c) for c in circuits]
        final_operators_list = []
        for k, ops in enumerate(operators_for_circuits):
            p = numpy.sqrt(probabilities[k])
            final_operators_list += [p * op for op in ops]
        return final_operators_list

    def noise_circuit_to_kraus_operators(self, circuit):
        # first, convert the list of dict-based noises to actual lists of matrices
        operators_list = [self.qobj_instruction_to_matrices(instruction) for instruction in circuit]
        Id = numpy.array([[1,0], [0,1]])
        final_operators = [Id]
        for operators in operators_list:
            final_operators = [op[0].dot(op[1]) for op in itertools.product(final_operators, operators)]
        return final_operators

    def qobj_instruction_to_matrices(self, instruction):
        if instruction['name'] == 'kraus':
            kraus_operators = [self.qobj_matrix_to_numpy_matrix(m) for m in instruction['params']]
            return kraus_operators
        raise RuntimeError("Does not know how to handle noises of type {}".format(instruction['name']))

    def transform_qobj(self, transform_channel, qobj):
        result_qobj = copy.deepcopy(qobj)
        try:
            noise_list = result_qobj["config"]["noise_model"]["errors"]
        except KeyError:
            return result_qobj  # if we can't find noises, we don't change anything
        for noise in noise_list:
            noise_in_kraus_form = self.qobj_noise_to_kraus_operators(noise)
            probabilities = self.transform(transform_channel, noise_in_kraus_form)
            transformed_channel_kraus_operators = self.probability_list_to_matrices(probabilities, transform_channel)
            qobj_kraus_operators = [self.numpy_matrix_to_qobj_matrix(m) for m in transformed_channel_kraus_operators]
            noise['probabilities'] = [1.0]
            noise['instructions'] = [[{"name": "kraus", "qubits": [0], "params": qobj_kraus_operators}]]
        return result_qobj

    @staticmethod
    def qobj_matrix_to_numpy_matrix(m):
        return numpy.array([[entry[0] + entry[1] * 1j for entry in row] for row in m])

    @staticmethod
    def numpy_matrix_to_qobj_matrix(m):
        return [[[numpy.real(a), numpy.imag(a)] for a in row] for row in (numpy.array(m))]

    def probability_list_to_matrices(self, probs, transform_channel):
        # probs are either a list or a dict of pairs of the form (probability, matrix_list)
        if isinstance(probs, dict):
            probs = probs.values()
        transformation_data = zip(probs, self.transform_channel_operators)
        # now assume probs is a list of pairs of the form (probability, matrix_list)
        id_matrix = numpy.eye(2) * numpy.sqrt(1 - sum(probs))
        return [id_matrix] + [numpy.sqrt(prob) * matrix for (prob, matrix_list) in transformation_data for matrix in matrix_list]

    # methods relevant to the transformation to quadratic programming instance

    @staticmethod
    def fidelity(channel):
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

    def single_qubit_full_clifford_group(self):
        # easy representation of all 23 non-identity 1-qubit Clifford group representation by using only S,H,X,Y,Z
        clifford_ops_names = [
            "X", "Y", "Z",
            "S", "SX", "SY", "SZ",
            "SH", "SHX", "SHY", "SHZ",
            "SHS", "SHSX", "SHSY", "SHSZ",
            "SHSH", "SHSHX", "SHSHY", "SHSHZ",
            "SHSHS", "SHSHSX", "SHSHSY", "SHSHSZ",
        ]
        base_ops = {
            'X': numpy.array([[0, 1], [1, 0]]),
            'Y': numpy.array([[0, -1j], [1j, 0]]),
            'Z': numpy.array([[1, 0], [0, -1]]),
            'S': numpy.array([[1, 0], [0, 1j]]),
            'H': numpy.sqrt(2) * numpy.array([[1, 1], [1, -1]])
        }
        return dict([(name, self.op_name_to_matrix(name, base_ops)) for name in clifford_ops_names])

    def op_name_to_matrix(self, name, base_ops):
        m = numpy.array([[1, 0], [0, 1]])
        for op_name in name:
            m = m * base_ops[op_name]
        return m