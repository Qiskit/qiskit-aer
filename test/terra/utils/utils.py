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
Utils
"""

from math import log2


def list2dict(counts_list, hex_counts=True):
    """Convert a list of counts to a dict"""
    if hex_counts:
        return {hex(i): val for i, val in enumerate(counts_list) if val > 0}
    # For bit-string counts we need to know number of qubits to
    # pad bitstring
    n_qubits = int(log2(counts_list))
    counts_dict = {}
    for i, val in enumerate(counts_list):
        if val > 0:
            key = bin(i)[2:]
            key = (n_qubits - len(key)) * "0" + key
            counts_dict[key] = val
    return counts_dict
