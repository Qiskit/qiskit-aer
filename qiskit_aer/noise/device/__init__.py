# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Device noise models module Qiskit Aer."""

from .device_models import depolarizing_noise_model
from .device_models import thermal_relaxation_noise_model
from .device_models import one_qubit_depolarizing_errors
from .device_models import one_qubit_thermal_relaxation_errors
from .device_models import one_qubit_readout_errors
from .device_models import two_qubit_depolarizing_errors
from .device_models import two_qubit_thermal_relaxation_errors
