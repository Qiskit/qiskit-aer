"""
Independent/manual construction and solving of DEs for verification of pulse simulator.
"""

import numpy as np
from scipy.linalg import expm
from qiskit.providers.aer.pulse.de.DE_Methods import ScipyODE
from qiskit.providers.aer.pulse.de.DE_Options import DE_Options

I = np.eye(2, dtype=complex)
X = np.array([[0., 1.], [1., 0.]])
Y = np.array([[0., -1j], [1j, 0.]])
Z = np.array([[1., 0.], [0., -1.]])

def channel_values(channel_freqs, channel_samples, dt, t):
    """Computes value of channels with given frequencies, samples, sample size and current time.

    Args:
        channel_freqs (array): 1d array of channel frequencies
        channel_samples (array): 2d array of channel samples, the first index being time step and
                                 the second index indexing channel
        dt (float): size of each sample
        t (float): current time

    Returns:
        array: array of channel values at the given time
    """

    sample_idx = int(t // dt)
    if sample_idx >= len(channel_samples):
        sample_idx = len(channel_samples) - 1

    sample_vals = channel_samples[sample_idx]

    return np.real(sample_vals * np.exp(1j * 2 * np.pi * channel_freqs * t))

def generator(drift, control_ops, chan_vals):
    """Compute the generator
       g = drift + chan_vals[0] * control_ops[0] + ... + chan_vals[0] * control_ops[0]
    """

    return drift +  np.tensordot(chan_vals, control_ops, axes=1)

def generator_in_frame(drift, control_ops, chan_vals, diag_frame, t):
    """Get the generator in the frame specified by diag_frame

    Args:
        drift (array): 2d drift generator
        control_ops (array): 3d array representing a list of control operators
        chan_vals (array): 1d array of the same length as control_ops
        diag_frame (array): 1d array representing an already diagonalized frame operator
                            assumed to be purely imaginary
        t (float): time

    Returns:
        array: generator in the given frame
    """

    G = generator(drift - np.diag(diag_frame), control_ops, chan_vals)

    U = np.exp(diag_frame * t)
    U_inv = U.conj()

    return np.diag(U_inv) @ G @ np.diag(U)


def simulate_system(y0, drift, control_ops, channel_freqs, channel_samples, dt, diag_frame):
    """Simulate the DE y' = G(t) @ y, where G(t) = drift + a0(t) * A0 + ... + ak(t) Ak, where
    control_ops = [A0, ..., Ak], and the aj(t) are the values of the signals specified
    by channel_freqs, channel_samples, and dt

    Args:
        y0 (array): initial state
        drift (array): 2d drift generator
        control_ops (array): 3d array representing a list of control operators
        channel_freqs (array): 1d array of channel frequencies
        channel_samples (array): 2d array of channel samples, the first index being time step and
                                 the second index indexing channel
        dt (float): size of each sample
        diag_frame (array): 1d array representing an already diagonalized frame operator
                            assumed to be purely imaginary

    Returns:
        array: final state of the DE
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
    """Simulate a basic 1 qubit model H(t) = 2 pi q_freq Z / 2 + 2 pi r D(t) * X / 2,
    where D(t) is the drive signal given by drive_freqs, drive_samples and dt
    """

    drift = -1j * 2 * np.pi * q_freq * Z / 2
    control_ops = -1j * np.array([ 2 * np.pi * r * X / 2 ])

    frame_op = -1j * 2 * np.pi * drive_freqs[0] * np.array([1., -1.]) / 2

    return simulate_system(y0, drift, control_ops, drive_freqs, drive_samples, dt, frame_op)

def simulate_2q_exchange_model(y0, q_freqs, r, j, drive_freqs, drive_samples, dt):
    """Simulate a basic 2 qubit model
        H(t) = 2 pi q_freq[0] Z0 / 2 + 2 pi r D0(t) * X0 / 2
               + 2 pi q_freq[1] Z1 / 2 + 2 pi r D1(t) * X1 / 2
               + 2 pi j (I0I1 + X0X1 + Y0Y1 + Z0Z1) / 2
    where D0(t) and D1(t) are the drive signals given by drive_freqs, drive_samples and dt
    """

    ZI_diag = np.kron(np.array([1., -1.]), np.array([1., 1.]))
    IZ_diag = np.kron(np.array([1., 1.]), np.array([1., -1.]))
    XI = np.kron(X, I)
    IX = np.kron(I, X)

    HI = (np.kron(I, I) + np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)) / 2

    drift_diag = -1j * 2 * np.pi * (q_freqs[0] * IZ_diag + q_freqs[1] * ZI_diag) / 2
    drift_diag = drift_diag - 1j * 2 * np.pi * j * np.diag(HI)
    drift = np.diag(drift_diag) - 1j * 2 * np.pi * j * (HI - np.diag(np.diag(HI)))

    control_ops = -1j * 2 * np.pi * r * np.array([IX, XI]) / 2

    return simulate_system(y0, drift, control_ops, drive_freqs, drive_samples, dt, drift_diag)

def simulate_3d_oscillator_model(y0, osc_freq, anharm, r, drive_freqs, drive_samples, dt):
    """Simulate a basic duffing odscillator model truncated at 3 dimensions, with
        H(t) = 2 pi osc_freq[0] a^\dagger a + 2 pi anharm (a^\dagger a)(a^dagger a - 1)
               + 2 pi r D(t) (a + a^\dagger)
    where D(t) is the drive signal given by drive_freqs, drive_samples and dt
    """

    drift_diag = -1j * (2 * np.pi * osc_freq * np.array([0., 1., 2.]) +
                        np.pi * anharm * np.array([0., 0., 2.]))

    drift = np.diag(drift_diag)
    osc_X = np.array([[0., 1., 0.],
                      [1., 0., np.sqrt(2)],
                      [0., np.sqrt(2), 0.]])
    control_ops = -1j * np.array([ 2 * np.pi * r * osc_X ])

    return simulate_system(y0, drift, control_ops, drive_freqs, drive_samples, dt, drift_diag)
