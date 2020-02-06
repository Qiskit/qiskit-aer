# -*- coding: utf-8 -*-
#!python
#cython: language_level = 3
#distutils: language = c++

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

cdef complex chan_value(double t,
                        unsigned int chan_num,
                        double freq_ch,
                        double[::1] chan_pulse_times,
                        complex[::1] pulse_array,
                        unsigned int[::1] pulse_ints,
                        double[::1] fc_array,
                        unsigned char[::1] register)
