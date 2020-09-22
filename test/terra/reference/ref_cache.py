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
Module for retrieving cached qobjs for testing.
"""

import pickle
from os import path

_CACHE_DIR = path.dirname(path.abspath(__file__))

def get_obj(fn):
    """Retrieve a cached qobj by filename"""
    ffn = path.join(_CACHE_DIR, 'cached', fn + ".pkl")
    with open(ffn, 'rb') as f:
        obj = pickle.load(f)
    return obj
