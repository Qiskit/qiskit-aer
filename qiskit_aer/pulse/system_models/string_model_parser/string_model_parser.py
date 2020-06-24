# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Parser for the string specification of Hamiltonians and noise models"""

import re
import copy
from collections import namedtuple, OrderedDict
import numpy as np
from .apply_str_func_to_qobj import apply_func
from .qobj_from_string import gen_oper


Token = namedtuple('Token', ('type', 'name'))

ham_elements = OrderedDict(
    QubOpr=re.compile(r"(?P<opr>O|Sp|Sm|X|Y|Z|I)(?P<idx>[0-9]+)"),
    PrjOpr=re.compile(r"P(?P<idx>[0-9]+),(?P<ket>[0-9]+),(?P<bra>[0-9]+)"),
    CavOpr=re.compile(r"(?P<opr>A|C|N)(?P<idx>[0-9]+)"),
    Func=re.compile(r"(?P<name>[a-z]+)\("),
    Ext=re.compile(r"\.(?P<name>dag)"),
    Var=re.compile(r"[a-z]+[0-9]*"),
    Num=re.compile(r"[0-9.]+"),
    MathOrd0=re.compile(r"[*/]"),
    MathOrd1=re.compile(r"[+-]"),
    BrkL=re.compile(r"\("),
    BrkR=re.compile(r"\)")
)


class HamiltonianParser:
    """ Generate QuTip hamiltonian object from string
    """
    def __init__(self, h_str, dim_osc, dim_qub):
        """ Create new quantum operator generator

        Parameters:
            h_str (list): list of Hamiltonian string
            dim_osc (dict): dimension of oscillator subspace
            dim_qub (dict): dimension of qubit subspace
        """
        self.h_str = h_str
        self.dim_osc = dim_osc
        self.dim_qub = dim_qub
        self.__td_hams = []
        self.__tc_hams = []
        self.__str2qopr = {}

    @property
    def compiled(self):
        """ Return Hamiltonian in OpenPulse handler format
        """
        return self.__tc_hams + self.__td_hams

    def parse(self, qubit_list=None):
        """ Parse and generate quantum class object
        """
        self.__td_hams = []
        self.__tc_hams = []

        # expand sum
        self._expand_sum()

        # convert to reverse Polish notation
        for ham in self.h_str:
            if len(re.findall(r"\|\|", ham)) > 1:
                raise Exception("Multiple time-dependent terms in %s" % ham)
            p_td = re.search(r"(?P<opr>[\S]+)\|\|(?P<ch>[\S]+)", ham)

            # find time-dependent term
            if p_td:
                coef, token = self._tokenizer(p_td.group('opr'), qubit_list)
                if token is None:
                    continue
                # combine coefficient to time-dependent term
                if coef:
                    td = '*'.join([coef, p_td.group('ch')])
                else:
                    td = p_td.group('ch')
                token = self._shunting_yard(token)
                _td = self._token2qobj(token), td

                self.__td_hams.append(_td)
            else:
                coef, token = self._tokenizer(ham, qubit_list)
                if token is None:
                    continue
                token = self._shunting_yard(token)

                if (coef == '') or (coef is None):
                    coef = '1.'

                _tc = self._token2qobj(token), coef

                self.__tc_hams.append(_tc)

    def _expand_sum(self):
        """ Takes a string-based Hamiltonian list and expands the _SUM action items out.
        """
        sum_str = re.compile(r"_SUM\[(?P<itr>[a-z]),(?P<l>[a-z\d{}+-]+),(?P<u>[a-z\d{}+-]+),")
        brk_str = re.compile(r"]")

        ham_list = copy.copy(self.h_str)
        ham_out = []

        while any(ham_list):
            ham = ham_list.pop(0)
            p_sums = list(sum_str.finditer(ham))
            p_brks = list(brk_str.finditer(ham))
            if len(p_sums) != len(p_brks):
                raise Exception('Missing correct number of brackets in %s' % ham)

            # find correct sum-bracket correspondence
            if any(p_sums) == 0:
                ham_out.append(ham)
            else:
                itr = p_sums[0].group('itr')
                _l = int(p_sums[0].group('l'))
                _u = int(p_sums[0].group('u'))
                for ii in range(len(p_sums) - 1):
                    if p_sums[ii + 1].end() > p_brks[ii].start():
                        break
                else:
                    ii = len(p_sums) - 1

                # substitute iterator value
                _temp = []
                for kk in range(_l, _u + 1):
                    trg_s = ham[p_sums[0].end():p_brks[ii].start()]
                    # generate replacement pattern
                    pattern = {}
                    for p in re.finditer(r"\{(?P<op_str>[a-z0-9*/+-]+)\}", trg_s):
                        if p.group() not in pattern:
                            sub = parse_binop(p.group('op_str'), operands={itr: str(kk)})
                            if sub.isdecimal():
                                pattern[p.group()] = sub
                            else:
                                pattern[p.group()] = "{%s}" % sub
                    for key, val in pattern.items():
                        trg_s = trg_s.replace(key, val)
                    _temp.append(''.join([ham[:p_sums[0].start()],
                                          trg_s, ham[p_brks[ii].end():]]))
                ham_list.extend(_temp)

        self.h_str = ham_out

        return ham_out

    def _tokenizer(self, op_str, qubit_list=None):
        """ Convert string to token and coefficient
        Check if the index is in qubit_list
        """

        # generate token
        _op_str = copy.copy(op_str)
        token_list = []
        prev = 'none'
        while any(_op_str):
            for key, parser in ham_elements.items():
                p = parser.match(_op_str)
                if p:
                    # find quantum operators
                    if key in ['QubOpr', 'CavOpr']:
                        _key = key
                        _name = p.group()
                        if p.group() not in self.__str2qopr.keys():
                            idx = int(p.group('idx'))
                            if qubit_list is not None and idx not in qubit_list:
                                return 0, None
                            name = p.group('opr')
                            opr = gen_oper(name, idx, self.dim_osc, self.dim_qub)
                            self.__str2qopr[p.group()] = opr
                    elif key == 'PrjOpr':
                        _key = key
                        _name = p.group()
                        if p.group() not in self.__str2qopr.keys():
                            idx = int(p.group('idx'))
                            name = 'P'
                            lvs = int(p.group('ket')), int(p.group('bra'))
                            opr = gen_oper(name, idx, self.dim_osc, self.dim_qub, lvs)
                            self.__str2qopr[p.group()] = opr
                    elif key in ['Func', 'Ext']:
                        _name = p.group('name')
                        _key = key
                    elif key == 'MathOrd1':
                        _name = p.group()
                        if prev not in ['QubOpr', 'PrjOpr', 'CavOpr', 'Var', 'Num']:
                            _key = 'MathUnitary'
                        else:
                            _key = key
                    else:
                        _name = p.group()
                        _key = key
                    token_list.append(Token(_key, _name))
                    _op_str = _op_str[p.end():]
                    prev = _key
                    break
            else:
                raise Exception('Invalid input string %s is found' % op_str)

        # split coefficient
        coef = ''
        if any([k.type == 'Var' for k in token_list]):
            for ii, _ in enumerate(token_list):
                if token_list[ii].name == '*':
                    if all([k.type != 'Var' for k in token_list[ii + 1:]]):
                        coef = ''.join([k.name for k in token_list[:ii]])
                        token_list = token_list[ii + 1:]
                        break
            else:
                raise Exception('Invalid order of operators and coefficients in %s' % op_str)

        return coef, token_list

    def _shunting_yard(self, token_list):
        """ Reformat token to reverse Polish notation
        """
        stack = []
        queue = []
        while any(token_list):
            token = token_list.pop(0)
            if token.type in ['QubOpr', 'PrjOpr', 'CavOpr', 'Num']:
                queue.append(token)
            elif token.type in ['Func', 'Ext']:
                stack.append(token)
            elif token.type in ['MathUnitary', 'MathOrd0', 'MathOrd1']:
                while stack and math_priority(token, stack[-1]):
                    queue.append(stack.pop(-1))
                stack.append(token)
            elif token.type in ['BrkL']:
                stack.append(token)
            elif token.type in ['BrkR']:
                while stack[-1].type not in ['BrkL', 'Func']:
                    queue.append(stack.pop(-1))
                    if not any(stack):
                        raise Exception('Missing correct number of brackets')
                pop = stack.pop(-1)
                if pop.type == 'Func':
                    queue.append(pop)
            else:
                raise Exception('Invalid token %s is found' % token.name)

        while any(stack):
            queue.append(stack.pop(-1))

        return queue

    def _token2qobj(self, tokens):
        """ Generate quantum class object from tokens
        """
        stack = []
        for token in tokens:
            if token.type in ['QubOpr', 'PrjOpr', 'CavOpr']:
                stack.append(self.__str2qopr[token.name])
            elif token.type == 'Num':
                stack.append(float(token.name))
            elif token.type in ['MathUnitary']:
                if token.name == '-':
                    stack.append(-stack.pop(-1))
            elif token.type in ['MathOrd0', 'MathOrd1']:
                op2 = stack.pop(-1)
                op1 = stack.pop(-1)
                if token.name == '+':
                    stack.append(op1 + op2)
                elif token.name == '-':
                    stack.append(op1 - op2)
                elif token.name == '*':
                    stack.append(op1 * op2)
                elif token.name == '/':
                    stack.append(op1 / op2)
            elif token.type in ['Func', 'Ext']:
                stack.append(apply_func(token.name, stack.pop(-1)))
            else:
                raise Exception('Invalid token %s is found' % token.name)

        if len(stack) > 1:
            raise Exception('Invalid mathematical operation in ' % tokens)

        return stack[0]


class NoiseParser:
    """ Generate QuTip noise object from dictionary
    Qubit noise is given in the format of nested dictionary:
        "qubit": {
            "0": {
                "Sm": 0.006
            }
        }
    and oscillator noise is given in the format of nested dictionary:
        "oscillator": {
            "n_th": {
                "0": 0.001
            },
            "coupling": {
                "0": 0.05
            }
        }
    these configurations are combined in the same dictionary
    """
    def __init__(self, noise_dict, dim_osc, dim_qub):
        """ Create new quantum operator generator

        Parameters:
            noise_dict (dict): dictionary of noise configuration
            dim_osc (dict): dimension of oscillator subspace
            dim_qub (dict): dimension of qubit subspace
        """
        self.noise_osc = noise_dict.get('oscillator', {'n_th': {}, 'coupling': {}})
        self.noise_qub = noise_dict.get('qubit', {})
        self.dim_osc = dim_osc
        self.dim_qub = dim_qub
        self.__c_list = []

    @property
    def compiled(self):
        """ Return noise configuration in OpenPulse handler format
        """
        return self.__c_list

    def parse(self):
        """ Parse and generate quantum class object
        """
        # Qubit noise
        for index, config in self.noise_qub.items():
            for opname, coef in config.items():
                # TODO: support noise in multi-dimensional system
                # TODO: support noise with math operation
                if opname in ['X', 'Y', 'Z', 'Sp', 'Sm']:
                    opr = gen_oper(opname, int(index), self.dim_osc, self.dim_qub)
                else:
                    raise Exception('Unsupported noise operator %s is given' % opname)
                self.__c_list.append(np.sqrt(coef) * opr)
        # Oscillator noise
        ndic = self.noise_osc['n_th']
        cdic = self.noise_osc['coupling']
        for (n_ii, n_coef), (c_ii, c_coef) in zip(ndic.items(), cdic.items()):
            if n_ii == c_ii:
                if c_coef > 0:
                    opr = gen_oper('A', int(n_ii), self.dim_osc, self.dim_qub)
                    if n_coef:
                        self.__c_list.append(np.sqrt(c_coef * (1 + n_coef)) * opr)
                        self.__c_list.append(np.sqrt(c_coef * n_coef) * opr.dag())
                    else:
                        self.__c_list.append(np.sqrt(c_coef) * opr)
            else:
                raise Exception('Invalid oscillator index in noise dictionary.')


def math_priority(o1, o2):
    """ Check priority of given math operation
    """
    rank = {'MathUnitary': 2, 'MathOrd0': 1, 'MathOrd1': 0}
    diff_ops = rank.get(o1.type, -1) - rank.get(o2.type, -1)

    if diff_ops > 0:
        return False
    else:
        return True


# pylint: disable=dangerous-default-value
def parse_binop(op_str, operands={}, cast_str=True):
    """ Calculate binary operation in string format
    """
    oprs = OrderedDict(
        sum=r"(?P<v0>[a-zA-Z0-9]+)\+(?P<v1>[a-zA-Z0-9]+)",
        sub=r"(?P<v0>[a-zA-Z0-9]+)\-(?P<v1>[a-zA-Z0-9]+)",
        mul=r"(?P<v0>[a-zA-Z0-9]+)\*(?P<v1>[a-zA-Z0-9]+)",
        div=r"(?P<v0>[a-zA-Z0-9]+)\/(?P<v1>[a-zA-Z0-9]+)",
        non=r"(?P<v0>[a-zA-Z0-9]+)"
    )

    for key, regr in oprs.items():
        p = re.match(regr, op_str)
        if p:
            val0 = operands.get(p.group('v0'), p.group('v0'))
            if key == 'non':
                # substitution
                retv = val0
            else:
                val1 = operands.get(p.group('v1'), p.group('v1'))
                # binary operation
                if key == 'sum':
                    if val0.isdecimal() and val1.isdecimal():
                        retv = int(val0) + int(val1)
                    else:
                        retv = '+'.join([str(val0), str(val1)])
                elif key == 'sub':
                    if val0.isdecimal() and val1.isdecimal():
                        retv = int(val0) - int(val1)
                    else:
                        retv = '-'.join([str(val0), str(val1)])
                elif key == 'mul':
                    if val0.isdecimal() and val1.isdecimal():
                        retv = int(val0) * int(val1)
                    else:
                        retv = '*'.join([str(val0), str(val1)])
                elif key == 'div':
                    if val0.isdecimal() and val1.isdecimal():
                        retv = int(val0) / int(val1)
                    else:
                        retv = '/'.join([str(val0), str(val1)])
                else:
                    retv = 0
            break
    else:
        raise Exception('Invalid string %s' % op_str)

    if cast_str:
        return str(retv)
    else:
        return retv
