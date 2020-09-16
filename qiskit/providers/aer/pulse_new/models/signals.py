# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np
from matplotlib import pyplot as plt

class Signal:
    """The most general mixed signal type, represented by a callable envelope function and a
    carrier frequency.
    """

    def __init__(self, envelope, carrier_freq=0.):
        self.envelope = envelope
        self.carrier_freq = carrier_freq

    def envelope_value(self, t):
        return self.envelope(t)

    def value(self, t):
        return self.envelope_value(t) * np.exp(1j * 2 * np.pi * self.carrier_freq * t)

    def conjugate(self):
        """Return a new signal that is the complex conjugate of this one"""
        return Signal(lambda t: np.conjugate(self.envelope_value(t)), -self.carrier_freq)

    def __mul__(self, other):
        return signal_mult(self, other)

    def __rmul__(self, other):
        return signal_mult(self, other)

    def __add__(self, other):
        return signal_add(self, other)

    def __radd__(self, other):
        return signal_add(self, other)

    def plot(self, t0, tf, N):
        x_vals = np.linspace(t0, tf, N)

        sig_vals = []
        for x in x_vals:
            sig_vals.append(self.value(x))

        plt.plot(x_vals, np.real(sig_vals))
        plt.plot(x_vals, np.imag(sig_vals))

    def plot_envelope(self, t0, tf, N):
        x_vals = np.linspace(t0, tf, N)

        sig_vals = []
        for x in x_vals:
            sig_vals.append(self.envelope_value(x))

        plt.plot(x_vals, np.real(sig_vals))
        plt.plot(x_vals, np.imag(sig_vals))


class Constant(Signal):
    """Constant.
    """

    def __init__(self, value):
        self._value = value
        self.carrier_freq = 0.

    def envelope_value(self, t=0.):
        return self._value

    def value(self, t=0.):
        return self.envelope_value()

    def conjugate(self):
        return Constant(np.conjugate(self._value))

    def __repr__(self):
        return 'Constant(' + repr(self._value) + ')'

class ConstantSignal(Signal):
    """A signal with constant envelope value but potentially non-zero carrier frequency."""

    def __init__(self, value, carrier_freq=0.):
        self._value = value
        self.carrier_freq = carrier_freq

    def envelope_value(self, t=0.):
        return self._value

    def conjugate(self):
        return ConstantSignal(np.conjugate(self._value), -self.carrier_freq)

    def __repr__(self):
        return ('ConstantSignal(value=' + repr(self._value) + ', carrier_freq=' +
                repr(self.carrier_freq) + ')')

class PiecewiseConstant(Signal):

    def __init__(self, dt, samples, start_time=None, duration=None, carrier_freq=0):

        self._dt = dt

        if samples is not None:
            self._samples = [_ for _ in samples]
        else:
            self._samples = [0] * duration

        if start_time is None:
            self._start_time = 0
        else:
            self._start_time = start_time

        self.carrier_freq = carrier_freq

    @property
    def duration(self) -> int:
        """
        Returns:
            duration: The duration of the signal in samples.
        """
        return len(self._samples)

    @property
    def dt(self) -> float:
        """
        Returns:
             dt: the duration of each sample.
        """
        return self._dt

    def envelope_value(self, t):
        if t < self._start_time * self._dt:
            return 0.0j

        idx = int(t // self._dt)

        # if the index is beyond the final time, return 0
        if idx >= self.duration:
            return 0.0j

        return self._samples[idx]

    def conjugate(self):
        return PiecewiseConstant(dt=self._dt,
                                 samples=np.conjugate(self._samples),
                                 start_time=self._start_time,
                                 duration=self.duration,
                                 carrier_freq=self.carrier_freq)


def signal_mult(sig1, sig2):
    """helper function for multiplying two signals together"""
    # ensure both arguments are signals
    if type(sig1) in [int, float, complex]:
        sig1 = Constant(sig1)
    if type(sig2) in [int, float, complex]:
        sig2 = Constant(sig2)

    # special cases to preserve specialized type
    if isinstance(sig1, Constant) and isinstance(sig1, Constant):
        return Constant(sig1._value * sig2._value)
    elif isinstance(sig1, ConstantSignal) and isinstance(sig2, ConstantSignal):
        return ConstantSignal(sig1._value * sig2._value, sig1.carrier_freq + sig2.carrier_freq)

    # if no special cases apply, simply multiply them as arbitrary time dependent functions
    new_carrier = sig1.carrier_freq + sig2.carrier_freq
    new_f = lambda t: sig1.envelope_value(t) * sig2.envelope_value(t)
    return Signal(new_f, new_carrier)

def signal_add(sig1, sig2):

    # ensure both arguments are signals
    if type(sig1) in [int, float, complex]:
        sig1 = Constant(sig1)
    if type(sig2) in [int, float, complex]:
        sig2 = Constant(sig2)

    if isinstance(sig1, Constant) and isinstance(sig2, Constant):
        return Constant(sig1._value + sig2._value)
    else:
        # if carrier freqs are the same we can combine the signal envelopes
        if sig1.carrier_freq == sig2.carrier_freq:
            # special cases
            if isinstance(sig1, ConstantSignal) and isinstance(sig2, ConstantSignal):
                return ConstantSignal(sig1._value + sig2._value, sig1.carrier_freq)
            elif isinstance(sig1, PiecewiseConstant) and isinstance(sig2, PiecewiseConstant):
                if sig1._dt == sig2._dt and sig1._start_time == sig2._start_time and sig1.duration == sig2.duration:
                    return PiecewiseConstant(dt=sig1._dt,
                                             samples=(sig1._samples + sig2._samples),
                                             start_time=sig1._start_time,
                                             duration=sig1.duration,
                                             carrier_freq=sig1.carrier_freq)
            # could add - ConstantSignal and PiecewiseConstant


            # if no special cases apply, simply add the envelopes together as functions
            return Signal(lambda t: sig1.envelope_value(t) + other.envelope_value(t),
                          self.carrier_freq)
        else:
            return Signal(lambda t: sig1.value(t) + sig2.value(t))
