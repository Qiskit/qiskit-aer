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
Helper functions for the pulse simulator
"""

from ..openpulse.qobj.digest import digest_pulse_obj

def compute_lo_freqs_from_hamiltonian(qobj=None,
                                      backend_options=None,
                                      noise_model=None):
    """Digest the pulse qobj and determine the qubit_lo_freq from
    the hamiltonian"""
    be_options_copy = backend_options.copy()
    be_options_copy['qubit_lo_freq'] = 'from_hamiltonian'
    openpulse_system = digest_pulse_obj(qobj, be_options_copy, noise_model)

    return openpulse_system.freqs
