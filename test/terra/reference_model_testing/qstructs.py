'''
This file contains data structures useful in Quantum Information.
Examples are: QuantumState, DensityMatrix, UnitaryOperator.
Also contains useful functions, for example randcomplex.
'''

import numbers
import math
import numpy as np


# ** is_square_matrix **
def is_square_matrix(mat):
    '''
    Checks if a numpy array represnts a square matrix
    '''
    s = np.shape(mat)
    if len(s)!=2:
        return False
    
    return s[0]==s[1]


# ** phase **
def phase(angle):
    '''
    Returns e^(i*angle)
    '''
    return math.e**(np.complex(0,1)*angle)


# ** is_close **
def is_close(a, b, rel_tol = 1e-09, abs_tol = 1e-09):
    """
    Returns true is 'a' and 'b' are close to each other.
    'a' and 'b' can be numbers of any type (integer, float, complex)
    and arrays of numbers.

    We use the existing function math.is_close

    numpy has a function for arrays, numpy.is_close,
    but it is not working well,
    therefore we will not call it.
    https://stackoverflow.com/questions/48156460/isclose-function-in-numpy-is-different-from-math
    """
    
    if isinstance(a, numbers.Number):
        
        if isinstance(a, np.complex):
            c = a
        else:
            c = np.complex(a, 0)

        if isinstance(b, np.complex):
            d = b
        else:
            d = np.complex(b, 0)
        
        return (math.isclose(np.real(c), np.real(d), rel_tol = rel_tol, abs_tol = abs_tol) and
                math.isclose(np.imag(c), np.imag(d), rel_tol = rel_tol, abs_tol = abs_tol))
    else:
        return all([is_close(x, y, rel_tol, abs_tol) for (x,y) in zip(np.array(a), np.array(b))])


# ** is_close_to_id **
def is_close_to_id(mat):
    """
    Checks if a matrix of complex numbers is equal (up to numerical fluctuations) to the identity matrix
    """
    if is_square_matrix(mat) == False:
        return false
        
    return is_close(mat, np.identity(len(mat)))


# ** randcomplex **
def randcomplex(n):
    """ Create a random vector of complex numbers """

    if float(n).is_integer() == False or n<1:
        raise ValueError('A request to generate a complex array of length ' + str(n) + '.' \
                         'Length must be an integer strictly greater than 0.')

    real_array = np.random.rand(n, 2)
    return np.array([complex(row[0], ((-1)**np.random.randint(2))*row[1]) for row in real_array ])


# ** arraydot **
def arraydot(a, b):
    """
    For two vectors of matrices a=[A_1,...,A_n] and b=[B_1,...,B_n],
    compute the sum of A_i*B_i over all i=1,...,n
    """
    return np.tensordot(a, b, axes=1)


# ** state_num2str
def state_num2str(basis_state_as_num, nqubits):
    return '{0:b}'.format(basis_state_as_num).zfill(nqubits)

# ** state_str2num
def state_str2num(basis_state_as_str):
    return int(basis_state_as_str, 2)

# ** state_array2str
def state_array2str(basis_state_as_array):
    return (''.join(map(str, basis_state_as_array)))

# ** state_str2array
def state_str2array(basis_state_as_str):
    return np.array(list(basis_state_as_str), dtype=int)

# ** state_num2array
def state_num2array(basis_state_as_num, nqubits):
    return state_str2array(state_num2str(basis_state_as_num, nqubits))

# ** state_array2num
def state_array2num(basis_state_as_array):
    return state_str2num(state_array2str(basis_state_as_array))

# ** state_reverse
def state_reverse(basis_state_as_num, nqubits):
    basis_state_as_str = state_num2str(basis_state_as_num, nqubits)
    new_str = basis_state_as_str[::-1]
    return state_str2num(new_str)


# ** get_extended_ops
def get_extended_ops(operators, qubits, nqubits):
    """
    Return the input operators, stretched over all qubits
    """

    # a list of all qubits
    all_qubits = np.array(range(nqubits))
    # a list of qubits not in "qubits", i.e.,
    # qubits that are not affected by the opertor
    diffset = np.setdiff1d(all_qubits, qubits)

    extended_ops = []
    for op in operators:
        new_op = np.zeros([2**nqubits, 2**nqubits], dtype=complex)

        for row in range(2**nqubits):
            for col in range(2**nqubits):
                
                row_as_array = state_num2array(row, nqubits)
                col_as_array = state_num2array(col, nqubits)

                if all(row_as_array[diffset] == col_as_array[diffset]) == False:
                    continue

                row_in_op_array = row_as_array[qubits]
                col_in_op_array = col_as_array[qubits]

                row_in_op = state_array2num(row_in_op_array)
                col_in_op = state_array2num(col_in_op_array)

                new_op[row][col] = op[row_in_op][col_in_op]

        extended_ops.append(new_op)

    return extended_ops


# ** ProbabilityDistribution **
class ProbabilityDistribution:
    """A vector of real non-negative numbers whose sum is 1"""
    
    def __init__(self, probs = None, n = None, not_normalized = None):
        """
        If probs is specified then this is the probability vector used for initialization,
        otherwise we create a random probability vector of length n.
        By default, probs is required to have sum 1.
        There is an option to set not_normalized to false,
        in which case we normalize the probability vector.
        """

        if n is not None and (float(n).is_integer() == False or n<1):
            raise ValueError('A request to generate a probability distribution of length ' + str(n) + '. ' \
                             'Length must be an integer strictly greater than 0.')
        
        if probs is None:
            # Will generate a random probability vector with the given length


            if not_normalized is not None:
                raise ValueError("Constructor of ProbabilityDistribution: " \
                                 "argument 'not_normalized' is irrelevant " \
                                 "if argument 'probs' is None")

            not_normalized = True

            if n is None:
                raise ValueError("Constructor of ProbabilityDistribution expects either " \
                                 "argument 'probs' or argument 'n' to be set.")
            
            # FIXME: give more thought about how to randomize the probability distribution
            before_normalization = np.random.rand(n)
            
        else:
            if not_normalized is None:
               not_normalized = False

            if isinstance(probs, numbers.Number):
                probs = [probs]

            if n is None:
                n = len(probs)
            else:
                if n != len(probs):
                    raise ValueError("Constructor of ProbabilityDistribution received " \
                                     "argument 'probs' with length " + str(len(probs)) + \
                                     " and argument 'n' equal to " + str(n) + ", " \
                                     "whereas it expects these two quntities to be equal to each other.")
            
            before_normalization = np.array(probs, dtype=float)
            
            if before_normalization.ndim != 1:
                raise ValueError('Constructor of ProbabilityDistribution received a vector of ' + \
                                 str(before_normalization.ndim) + 'dimensions, \
                                 whereas it expects exactly 1 dimension.')

        # math.fsum is probably more stable than np.sum
        if not_normalized == True:
            self.probs = before_normalization / math.fsum(before_normalization)
        else:
            self.probs = before_normalization

        if is_close(1, math.fsum(self.probs)) == False:
            raise ValueError('Probability vector is not normalized')


    def __len__(self):
        return len(self.probs)
            


# ** QuantumState **
class QuantumState:
    """A vector of complex numbers whose norm is 1"""

    def __init__(self, amplitudes = None, nqubits = None, not_normalized = None):
        """
        If amplitudes is specified then this is the amplitudes vector used for initialization,
        otherwise we create a random quantum state of length 2**nqubits.
        By default, the amplitudes vector is required to be normalized.
        There is an option to set not_normalized to false,
        in which case we normalize the amplitudes vector.
        """

        if amplitudes is None:
           # Will generate a random quantum state for the given number of qubits

           if nqubits is None:
               raise ValueError("Constructor of QuantumState: argument 'nqubits' cannot be None if argument 'amplitudes' is None")

           if not_normalized is not None:
               raise ValueError("Constructor of QuantumState: " \
                                "argument 'not_normalized' is irrelevant " \
                                "if argument 'amplitudes' is None")

           if float(nqubits).is_integer() == False or nqubits < 1:
               raise ValueError('A request to generate a quantum state with ' + str(n) + ' qubits. ' \
                                'Number of qubits must be an integer strictly greater than 0.')

           not_normalized = True

           # FIXME: give more thought about how to randomize the amplitudes
           before_normalization = randcomplex(2**nqubits)

        else:
           if not_normalized is None:
               not_normalized = False

           if nqubits is None:
               nqubits = math.log2(len(amplitudes))
               if nqubits.is_integer() == False:
                   raise ValueError('Constructor of QuantumState received an amplitudes vector ' \
                                    'of length ' + len(amplitudes) + \
                                    ', whereas the number of amplitudes must be a power of 2.')
           else:           
               if nqubits != math.log2(len(amplitudes)):
                   raise ValueError('Constructor of QuantumState received ' + \
                                    str(len(amplitudes)) + ' amplitudes for ' + \
                                    str(nqubits) + ' qubits, ' \
                                    'whereas it expects the number of amplitudes to be 2 to the power of the number of qubits.')

           before_normalization = np.array(amplitudes, dtype=complex)

           if before_normalization.ndim != 1:
               raise ValueError('Constructor of QuantumState received a vector of ' + \
                                str(before_normalization.ndim) + 'dimensions, \
                                whereas it expects exactly 1 dimension.')

        if not_normalized == True:
            self.amplitudes = before_normalization / np.linalg.norm(before_normalization)
        else:
            self.amplitudes = before_normalization

        if is_close(1, np.linalg.norm(self.amplitudes)) == False:
            raise ValueError('Quantum state is not normalized')
       
        self.nqubits = nqubits
        self.nstates = 2**nqubits


    @staticmethod
    def ground_state(nqubits):
        """
        Creates a ground state.
        """

        amplitudes = np.zeros(2**nqubits)
        amplitudes[0] = 1
        return QuantumState(amplitudes = amplitudes)


    def __str__(self):
        return str(self.amplitudes)
    
      

# ** DensityMatrix **
class DensityMatrix:

    def __init__(self, states = None, probs = None, mat = None):
        """
        Initialized either by a density matrix mat,
        which is required to be positive and have trace 1,
        or by a set of states with a correspoding set of probabilities for each state
        """

        # FIXME: accept a mixture of quantum states and density matrices,
        # recursively apply for all of them,
        # and then sum with the given weights.

        if mat is not None:
            if states is not None or probs is not None:
                raise ValueError("Constructor of DensityMatrix: " \
                                 "If argument 'mat' is not None " \
                                 "then arguments 'states' and 'probs' must be set to None")

            self.rho = np.array(mat, dtype=complex)

        else:
            if states is None:
                raise ValueError("Constructor of DensityMatrix: " \
                                 "If argument 'mat' is None " \
                                 "then argument 'states' cannot be set to None" )
            
            if isinstance(states, QuantumState):
                states = [states]

            if probs is None:
                probs = ProbabilityDistribution(1)
                
            if isinstance(states, list) == False or any(isinstance(s, QuantumState) == False for s in states):
                raise ValueError('Constructor of DensityMatrix expects a list of QuantumState')

            if isinstance(probs, ProbabilityDistribution) == False:
                raise ValueError('Constructor of DensityMatrix expects a probability vector')

            if len(probs) != len(states):
                raise ValueError('Constructor of DensityMatrix received ' \
                                 'a state vector of length ' + str(len(states)) + \
                                 ' and a probability distribution of length ' + str(len(probs)) + \
                                 ', whereas it expects both to be of the same length.')

            if any(s.nqubits != states[0].nqubits for s in states):
                raise ValueError('Constructor of DensityMatrix expects all quantum states to have the same number of qubits')

            mats = [np.outer(x, np.conj(x)) for x in [s.amplitudes for s in states]]
            self.rho = arraydot(probs.probs, mats)

        if len(self.rho) != len(self.rho[0]):
            raise ValueError('Constructor of DensityMatrix received a matrix with ' + \
                             len(self.rho) + ' rows and ' + len(self.rho[0]) + 'columns, ' \
                             'whereas it expects a sqaured matrix.')

        self.nstates = len(self.rho)
        self.nqubits = math.log2(self.nstates)

        if self.nqubits.is_integer() == False:
            raise ValueError('Constructor of DensityMatrix received a matrix with ' + \
                             self.nstates + ' rows and columns, ' \
                             'whereas the number of rows and columns must be a power of 2.')

        self.nqubits = int(self.nqubits)

        if is_close(1, np.trace(self.rho)) == False:
            raise ValueError('Constructor of DensityMatrix received a matrix with ' \
                             'trace ' + str(np.trace(self.rho)) + \
                             ', whereas the trace must be equal to 1.')

        if np.any(np.linalg.eigvals(self.rho) < -0.05):
            raise ValueError('Constructor of DensityMatrix expects a positive matrix.')


    def qop(self, operators):
        """
        Apply a set of operators on the density matrix, resulting in a new density matrix
        rho' = sum_k E_k rho E_k^\dagger
        """

        # FIXME: add an option to provide the operators as a set of unitary matrices with weights.
        
        return DensityMatrix(mat = sum( [np.dot(op, np.dot(self.rho, np.matrix(op).H)) for op in operators] ))
        

    def qop_on_qubits(self, qubits, operators):
        """
        Apply a set of operators on the specified qubits
        """    
        return self.qop(get_extended_ops(operators, qubits, self.nqubits))


    def observable(self, params):

        result = 0
        for component in params:
            mat = np.identity(2**self.nqubits, dtype=complex)
            for block in component[1]:
                extended_mat = get_extended_ops([block[1]], block[0], self.nqubits)[0]
                mat = np.dot(extended_mat, mat)
            result += component[0]*np.trace(np.dot(mat, self.rho))

        return result


    def reset(self, qubit):
        """
        Reset the specified qubit
        """

        new_rho = np.zeros([2**self.nqubits, 2**self.nqubits], dtype=complex)

        for row in range(2**self.nqubits):
            for col in range(2**self.nqubits):

                row_as_array = state_num2array(row, self.nqubits)
                col_as_array = state_num2array(col, self.nqubits)

                if row_as_array[qubit] == col_as_array[qubit]:
                    row_as_array[qubit] = 0
                    col_as_array[qubit] = 0

                    new_row = state_array2num(row_as_array)
                    new_col = state_array2num(col_as_array)

                    new_rho[new_row][new_col] += self.rho[row][col]

        return DensityMatrix(mat=new_rho)
    

    def extract_probs(self):
        """
        Specify the probability for each basis state.
        Format matches the simulator's output,
        meaning that the basis states are displayed in hexa,
        and more importantly qubit 0 is LSB.
        For example, for two qubits:
        {'0x0: 0.3, '0x1': 0.7}
        means probability 0.3 for a state where both qubits are 0,
        and probability 0.7 for a state where qubit 0 is 1 and qubit 1 is 0.
        """

        probs = dict()
        for basis_state_as_num in range(2**self.nqubits):

            rho_entry = self.rho[basis_state_as_num, basis_state_as_num]

            if is_close(0, rho_entry) == False:
                reverse_hex = hex(state_reverse(basis_state_as_num, self.nqubits))
                if reverse_hex in probs.keys():
                    probs[reverse_hex] += rho_entry
                else:
                    probs[reverse_hex] = rho_entry

        return probs


    def __str__(self):
        return str(self.rho)
    


# ** UnitaryOperation **
class UnitaryOperation:

    def __init__(self, mat = None, angles = None, n = None):
        """        
        Initialized in one of the following ways:
        - By a matrix mat, which is required to be unitary.
        - By the angles beta, gamma, and delta, in which case a 2x2 unitary matrix is generated,
          tensored by itself n-1 times, resulting in a (2^n)x(2^n) matrix.

        If 'mat' and 'angles' are both None,
        generates a (2^n)x(2^n) random unitary matrix,
        which is a tensor product of n 2x2 random unitary matrices.
        """

        if n is not None and n<1:
            raise ValueError("Constructor of UnitaryOperation: Matrix dimensions must be positive.")

        if mat is not None:
            if angles is not None:
                raise ValueError("Constructor of UnitaryOperation: " \
                                 "If argument 'mat' is not None " \
                                 "then argument 'angles' must be set to None")

            self.mat = np.array(mat, dtype=complex)
            if is_square_matrix(self.mat) == False:
                raise ValueError("Constructor of UnitaryOperation: " \
                                 "Parameter 'mat' must be a square matrix")
                   
            if n is not None:
                if any(x!=2**n for x in np.shape(self.mat)):
                    raise ValueError("Constructor of UnitaryOperation: Wrong matrix dimensions")
                                                    
        else:
            self.mat = np.array([1])

            if n is None:
                n = 1

            for i in range(n):

                if angles is None:
                    angles = np.random.rand(3)*math.pi
                else:
                    angles = np.array(angles)
                    if len(angles)!=3:
                        raise ValueError("There must be 3 angles for the Unitary operation")

                [beta, gamma, delta] = angles

                mat_beta = np.array([[phase(-beta/2), 0], [0, phase(beta/2)]])
                mat_gamma = np.array([[math.cos(gamma/2), -math.sin(gamma/2)], [math.sin(gamma/2), math.cos(gamma/2)]])
                mat_delta = np.array([[phase(-delta/2), 0], [0, phase(delta/2)]])

                mat = np.dot(mat_beta, np.dot(mat_gamma, mat_delta))
                self.mat = np.kron(mat, self.mat)

        if is_close_to_id(np.dot(np.matrix(self.mat).H, self.mat)) == False:
            raise ValueError('Constructor of UnitaryOperation: matrix is not unitary')
