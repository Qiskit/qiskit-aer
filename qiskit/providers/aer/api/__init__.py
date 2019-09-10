# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""On-Premise Simulator API connector."""

from .httpconnector import ApiError, BadBackendError, RegisterSizeError
from .httpconnector import HttpConnector
from .sshconnector import ApiError, BadBackendError, RegisterSizeError
from .sshconnector import SshConnector
