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
=============================================================
Device Noise Model (:mod:`qiskit.providers.aer.noise.device`)
=============================================================

.. currentmodule:: qiskit.providers.aer.noise.device

Approximate noise models for a hardware device can be generated from the
device properties using the functions in this module.

Basic device noise model
========================

The :func:`basic_device_noise_model` function generates a noise model
based on:

* 1 and 2 qubit gate errors consisting of a
  :func:`~qiskit.providers.aer.noise.errors.depolarizing_error` followed
  by a :func:`~qiskit.providers.aer.noise.errors.thermal_relaxation_error`.

* Single qubit :class:`~qiskit.providers.aer.noise.errors.ReadoutError` on
  all measurements.

The Error error parameters are tuned for each individual qubit based on
the :math:`T_1`, :math:`T_2`, frequency and readout error parameters for
each qubit, and the gate error and gate time parameters for each gate
obtained from the device backend properties.


Functions
=========

.. autosummary::
    :toctree: ../stubs/

    basic_device_noise_model
    basic_device_readout_errors
    basic_device_gate_errors


Helper functions
================

The following helper functions can be used to extract parameters from
a device ``BackendProperties`` object.

.. autosummary::
    :toctree: ../stubs/

    parameters.gate_param_values
    parameters.gate_error_values
    parameters.gate_length_values
    parameters.readout_error_values
    parameters.thermal_relaxation_values
"""

from .models import basic_device_noise_model
from .models import basic_device_readout_errors
from .models import basic_device_gate_errors
from . import parameters
