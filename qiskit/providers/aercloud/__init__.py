# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Backends provided by On-premise Simulator."""
from qiskit.qiskiterror import QiskitError
from .aercloudprovider import AerCloudProvider
from .aercloudbackend import AerCloudBackend
from .aercloudjob import AerCloudJob

# Global instance to be used as the entry point for convenience.
AerCloud = AerCloudProvider()

