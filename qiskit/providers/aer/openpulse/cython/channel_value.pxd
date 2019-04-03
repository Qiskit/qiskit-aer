# -*- coding: utf-8 -*-
#!python
#cython: language_level = 3
#distutils: language = c++

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

cdef complex channel_value(double t,
                           unsigned int chan_num,
                           double[::1] chan_pulse_times,
                           complex[::1] pulse_array,
                           unsigned int[::1] pulse_ints,
                           double[::1] fc_array,
                           unsigned char[::1] register)