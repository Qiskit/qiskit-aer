"""
Independent/manual construction and solving of DEs for verification of pulse simulator.
"""

import numpy as np
from scipy.linalg import expm
from qiskit.providers.aer.pulse.de.DE_Methods import ScipyODE
from qiskit.providers.aer.pulse.de.DE_Options import DE_Options

X = np.array([[0., 1.], [1., 0.]])
Y = np.array([[0., -1j], [1j, 0.]])
Z = np.array([[1., 0.], [0., -1.]])

def channel_values(channel_freqs, channel_samples, dt, t):

    sample_idx = int(t // dt)
    if sample_idx >= len(channel_samples):
        sample_idx = len(channel_samples) - 1

    sample_vals = channel_samples[sample_idx]

    return np.real(sample_vals * np.exp(1j * 2 * np.pi * channel_freqs * t))

def generator(drift, control_ops, chan_vals):
    return drift +  np.tensordot(chan_vals, control_ops, axes=1)

def generator_in_frame(drift, control_ops, chan_vals, diag_frame, t):
    """ Get the generator in the frame specified by diag_frame

    Args:
        drift (array): 2d drift generator
        control_ops (array): 3d array representing a list of control operators
        chan_vals (array): 1d array of the same length as control_ops
        diag_frame (array): 1d array representing an already diagonalized frame operator
                            assumed to be purely imaginary
        t (float): time
    """

    G = generator(drift - np.diag(diag_frame), control_ops, chan_vals)

    U = np.exp(diag_frame * t)
    U_inv = U.conj()

    return np.diag(U_inv) @ G @ np.diag(U)


def simulate_system(y0, drift, control_ops, channel_freqs, channel_samples, dt, diag_frame):
    """Simulate the DE y' = G(t) @ y, where G(t) = drift + a0(t) * A0 + ... + ak(t) Ak, where
       control_ops = [A0, ..., Ak], and the aj(t) are the values of the signals specified
       by channel_freqs and channel_samples
    """

    # if all channel freqs are 0 simulate using matrix exponentiation
    if all(channel_freqs == 0):

        yf = y0
        for t_idx in range(len(channel_samples)):
            yf = expm(generator(drift, control_ops, channel_samples[t_idx]) * dt) @ yf

        return yf

    # else, simulate using standard ODE solver
    else:
        # set up rhs function in frame
        def rhs(t, y):
            chan_vals = channel_values(channel_freqs, channel_samples, dt, t)
            gen = generator_in_frame(drift, control_ops, chan_vals, diag_frame, t)
            return gen @ y

        de_options = DE_Options(method='RK45')
        ode_method = ScipyODE(t0=0., y0=y0, rhs=rhs, options=de_options)

        T = len(channel_samples) * dt
        ode_method.integrate(T)
        yf = np.exp(diag_frame * T) * ode_method.y

        return yf

def simulate_1q_model(y0, q_freq, r, drive_freqs, drive_samples, dt):

    drift = -1j * 2 * np.pi * q_freq * Z / 2
    control_ops = -1j * np.array([ 2 * np.pi * r * X / 2 ])

    frame_op = -1j * 2 * np.pi * drive_freqs[0] * np.array([1., -1.]) / 2

    return simulate_system(y0, drift, control_ops, drive_freqs, drive_samples, dt, frame_op)


def simulate_3d_oscillator_model(y0, osc_freq, anharm, r, drive_freqs, drive_samples, dt):

    drift_diag = -1j * (2 * np.pi * osc_freq * np.array([0., 1., 2.]) +
                        np.pi * anharm * np.array([0., 0., 2.]))
    
    drift = np.diag(drift_diag)
    osc_X = np.array([[0., 1., 0.],
                      [1., 0., np.sqrt(2)],
                      [0., np.sqrt(2), 0.]])
    control_ops = -1j * np.array([ 2 * np.pi * r * osc_X ])

    return simulate_system(y0, drift, control_ops, drive_freqs, drive_samples, dt, drift_diag)
