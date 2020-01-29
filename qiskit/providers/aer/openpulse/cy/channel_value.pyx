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

cimport cython

cimport cython
from libc.math cimport floor, M_PI

cdef extern from "<complex>" namespace "std" nogil:
    double complex exp(double complex x)


@cython.cdivision(True)
cdef inline int get_arr_idx(double t, double start, double stop, int len_arr):
    """Computes the array index value for sampling a pulse in pulse_array.

    Args:
        t (double): The current simulation time.
        start (double): Start time of pulse in question.
        stop (double): Stop time of pulse.
        len_arr (int): Length of the pulse sample array.

    Returns:
        int: The array index value.
    """
    return <int>floor(((t-start)/(stop-start)*len_arr))

@cython.boundscheck(False)
cdef complex chan_value(double t,
                        unsigned int chan_num,
                        double freq_ch,
                        double[::1] chan_pulse_times,
                        complex[::1] pulse_array,
                        unsigned int[::1] pulse_ints,
                        double[::1] fc_array,
                        unsigned char[::1] register):
    """Computes the value of a given channel at time `t`.

    Args:
        t (double): Current time.
        chan_num (int): The int that labels the channel.
        chan_pulse_times (int array): Array containing
            start_time, stop_time, pulse_int, conditional for
            each pulse on the channel.
        pulse_array (complex array): The array containing all the
            pulse data in the passed pulse qobj.
        pulse_ints (int array): Array that tells you where to start
            indexing pulse_array for a given pulse labeled by
            chan_pulse_times[4*kk+2].
        current_pulse_idx (int array),
        freq_ch (doule) channel frequency:
    """
    cdef size_t kk
    cdef double start_time, stop_time, phase=0
    cdef int start_idx, stop_idx, offset_idx, temp_int, cond
    cdef complex out = 0
    # This is because each entry has four values:
    # start_time, stop_time, pulse_int, conditional
    cdef unsigned int num_times = chan_pulse_times.shape[0] // 4

    for kk in range(num_times):
        # the time is overlapped with the kkth pulse
        start_time = chan_pulse_times[4*kk]
        stop_time = chan_pulse_times[4*kk+1]
        if start_time <= t < stop_time:
            cond = <int>chan_pulse_times[4*kk+3]
            if cond < 0 or register[cond]:
                temp_int = <int>chan_pulse_times[4*kk+2]
                start_idx = pulse_ints[temp_int]
                stop_idx = pulse_ints[temp_int+1]
                offset_idx = get_arr_idx(t, start_time, stop_time, stop_idx-start_idx)
                out = pulse_array[start_idx+offset_idx]
            break
    # Compute the frame change up to time t
    if out != 0:
        num_times = fc_array.shape[0] // 3
        for kk in range(num_times):
            if t >= fc_array[3*kk]:
                do_fc = 1
                # Check if FC is conditioned on register
                if fc_array[3*kk+2] >= 0:
                    # If condition not satisfied no do FC
                    if not register[<int>fc_array[3*kk+2]]:
                        do_fc = 0
                if do_fc:
                    # Update the frame change value
                    phase += fc_array[3*kk+1]
            else:
                break
        if phase != 0:
            out *= exp(1j*phase)
        out *= exp(-1j*2*M_PI*freq_ch*t)
    return out
