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

"""Temporary folder for string parsers.

Currently:
- Nothing in this folder directly depends on QuTip, but it does depend on qutip
  functionality (i.e. it manipulates qutip qobjs, though never directly creates them
  itself)
- The connection to qutip is through direct_qutip_dependence/qobj_generators
    - The calls to this can be replaced with calls to something else that creates
      the instances of the operator class that we end up using
"""
