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
from typing import Optional, List, Callable, Union

import numpy as np
from matplotlib import pyplot as plt


class BaseSignal(ABC):

    @abstractmethod
    def conjugate(self):
        """Return a new signal that is the complex conjugate of self."""

    @abstractmethod
    def envelope_value(self, t: float) -> complex:
        """Evaluates the envelope at time t."""

    @abstractmethod
    def value(self, t: float) -> complex:
        """Return the value of the signal at time t."""

    def __mul__(self, other):
        return signal_multiply(self, other)

    def __rmul__(self, other):
        return signal_multiply(self, other)

    def __add__(self, other):
        return signal_add(self, other)

    def __radd__(self, other):
        return signal_add(self, other)


class Signal(BaseSignal):
    """The most general mixed signal type, represented by a callable envelope function and a
    carrier frequency.
    """

    def __init__(self, envelope: Callable, carrier_freq: float = 0.):
        """
        Initializes a signal given by an envelop and an optional carrier.

        Args:
            envelope: Envelope function of the signal.
            carrier_freq: Frequency of the carrier.
        """
        self.envelope = envelope
        self.carrier_freq = carrier_freq

    def envelope_value(self, t: float) -> complex:
        """Evaluates the envelope at time t."""
        return self.envelope(t)

    def value(self, t) -> complex:
        """Return the value of the signal at time t."""
        return self.envelope_value(t) * np.exp(1j * 2 * np.pi * self.carrier_freq * t)

    def conjugate(self):
        """Return a new signal that is the complex conjugate of this one"""
        return Signal(lambda t: self.envelope_value(t).conjugate(), -self.carrier_freq)

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


class Constant(BaseSignal):
    """
    Constant that can appear in the Hamiltonian.
    """

    def __init__(self, value: complex):
        self._value = value

    def envelope_value(self, t=0.) -> complex:
        return self._value

    def value(self, t=0.) -> complex:
        return self._value

    def conjugate(self):
        return Constant(self._value.conjugate())

    def __repr__(self):
        return 'Constant(' + repr(self._value) + ')'


class ConstantSignal(BaseSignal):
    """A signal with constant envelope value but potentially non-zero carrier frequency."""

    def __init__(self, value: complex, carrier_freq: float = 0.):
        self._value = value
        self.carrier_freq = carrier_freq

    def envelope_value(self, t: float = 0.):
        return self._value

    def value(self, t: float) -> complex:
        """Return the value of the signal at time t."""
        return self.envelope_value() * np.exp(1j * 2 * np.pi * self.carrier_freq * t)

    def conjugate(self):
        return ConstantSignal(self._value.conjugate(), -self.carrier_freq)

    def __repr__(self):
        return ('ConstantSignal(value=' + repr(self._value) + ', carrier_freq=' +
                repr(self.carrier_freq) + ')')


class PiecewiseConstant(BaseSignal):

    def __init__(self, dt: float, samples: Union[np.array, List], start_time: float = 0.,
                 duration: int = None, carrier_freq: float = 0):

        self._dt = dt

        if samples is not None:
            self._samples = [_ for _ in samples]
        else:
            self._samples = [0] * duration

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

    @property
    def samples(self) -> np.array:
        return np.array([_ for _ in self._samples])

    @property
    def start_time(self) -> float:
        return self._start_time

    def envelope_value(self, t: float) -> complex:
        if t < self._start_time * self._dt:
            return 0.0j

        idx = int((t - self._start_time) // self._dt)

        # if the index is beyond the final time, return 0
        if idx >= self.duration:
            return 0.0j

        return self._samples[idx]

    def value(self, t) -> complex:
        """Return the value of the signal at time t."""
        return self.envelope_value(t) * np.exp(1j * 2 * np.pi * self.carrier_freq * t)

    def conjugate(self):
        return PiecewiseConstant(dt=self._dt,
                                 samples=np.conjugate(self._samples),
                                 start_time=self._start_time,
                                 duration=self.duration,
                                 carrier_freq=self.carrier_freq)


def signal_multiply(sig1: Union[BaseSignal, float, int, complex], sig2: Union[BaseSignal, float, int, complex]):
    """helper function for multiplying two signals together."""

    # ensure both arguments are signals
    if type(sig1) in [int, float, complex]:
        sig1 = Constant(sig1)

    if type(sig2) in [int, float, complex]:
        sig2 = Constant(sig2)

    # Multiplications with Constant
    if isinstance(sig1, Constant) and isinstance(sig1, Constant):
        return Constant(sig1.value() * sig2.value())

    elif isinstance(sig1, Constant) and isinstance(sig2, ConstantSignal):
        return ConstantSignal(sig1.value() * sig2.envelope_value(), sig2.carrier_freq)

    elif isinstance(sig1, Constant) and isinstance(sig2, Signal):
        return Signal(lambda t: sig1.value() * sig2.envelope_value(t), sig2.carrier_freq)

    elif isinstance(sig1, Constant) and isinstance(sig2, PiecewiseConstant):
        return PiecewiseConstant(sig1.value()*sig2.samples, sig2.carrier_freq)

    # Multiplications with ConstantSignal
    elif isinstance(sig1, ConstantSignal) and isinstance(sig2, ConstantSignal):
        return ConstantSignal(sig1.envelope_value() * sig2.envelope_value(),
                              sig1.carrier_freq + sig2.carrier_freq)

    elif isinstance(sig1, ConstantSignal) and isinstance(sig2, Signal):
        return Signal(lambda t: sig1.value() * sig2.envelope_value(t), sig1.carrier_freq + sig2.carrier_freq)

    elif isinstance(sig1, ConstantSignal) and isinstance(sig2, PiecewiseConstant):
        new_samples = []
        for idx, sample in enumerate(sig2.samples):
            new_samples.append(sample * sig1.envelope_value(sig2.dt*idx + sig2.start_time))

        return PiecewiseConstant(sig2.dt, new_samples, sig1.carrier_freq + sig2.carrier_freq)

    # Multiplications with Signal
    elif isinstance(sig1, Signal) and isinstance(sig2, Signal):
        return Signal(lambda t: sig1.envelope_value() * sig2.envelope_value(t), sig1.carrier_freq + sig2.carrier_freq)

    elif isinstance(sig1, Signal) and isinstance(sig2, PiecewiseConstant):
        new_samples = []
        for idx, sample in enumerate(sig2.samples):
            new_samples.append(sample * sig1.envelope_value(sig2.dt*idx + sig2.start_time))

        return PiecewiseConstant(sig2.dt, new_samples, sig1.carrier_freq + sig2.carrier_freq)

    # Multiplications with PiecewiseConstant
    elif isinstance(sig1, PiecewiseConstant) and isinstance(sig2, PiecewiseConstant):
        # Assume sig2 always has the larger dt
        if sig1.dt > sig2.dt:
            sig1, sig2 = sig2, sig1

        new_samples = []
        for idx, sample in enumerate(sig1.samples):
            new_samples.append(sample * sig2.envelope_value(sig1.dt*idx + sig1.start_time))

        return PiecewiseConstant(sig1.dt, new_samples, sig1.carrier_freq + sig2.carrier_freq)

    # Other symmetric cases
    return signal_multiply(sig2, sig1)


def signal_add(sig1, sig2):

    # ensure both arguments are signals
    if type(sig1) in [int, float, complex]:
        sig1 = Constant(sig1)
    if type(sig2) in [int, float, complex]:
        sig2 = Constant(sig2)

    # Multiplications with Constant
    if isinstance(sig1, Constant) and isinstance(sig1, Constant):
        return Constant(sig1.value() + sig2.value())

    elif isinstance(sig1, Constant) and isinstance(sig2, ConstantSignal):
        raise NotImplementedError

    elif isinstance(sig1, Constant) and isinstance(sig2, Signal):
        raise NotImplementedError

    elif isinstance(sig1, Constant) and isinstance(sig2, PiecewiseConstant):
        raise NotImplementedError

    # Multiplications with ConstantSignal
    elif isinstance(sig1, ConstantSignal) and isinstance(sig2, ConstantSignal):
        raise NotImplementedError

    elif isinstance(sig1, ConstantSignal) and isinstance(sig2, Signal):
        raise NotImplementedError

    elif isinstance(sig1, ConstantSignal) and isinstance(sig2, PiecewiseConstant):
        raise NotImplementedError

    # Multiplications with Signal
    elif isinstance(sig1, Signal) and isinstance(sig2, Signal):
        raise NotImplementedError

    elif isinstance(sig1, Signal) and isinstance(sig2, PiecewiseConstant):
        raise NotImplementedError

    # Multiplications with PiecewiseConstant
    elif isinstance(sig1, PiecewiseConstant) and isinstance(sig2, PiecewiseConstant):
        raise NotImplementedError

    # Other symmetric cases
    return signal_multiply(sig2, sig1)

