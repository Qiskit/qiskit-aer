
# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name, bad-continuation

"""Provider for the local AER QV backend."""

import logging

from qiskit.backends import BaseProvider
from aer_qv_wrapper import AerQvSimulatorWrapper

logger = logging.getLogger(__name__)


class AerQvProvider(BaseProvider):
    """Provider for the local JKU backend."""

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

        # Populate the list of local AER QV backends.
        self.backends = {'local_qv_simulator': AerQvSimulatorWrapper()}

    def get_backend(self, name):
        return self.backends[name]

    def available_backends(self, filters=None):
        # pylint: disable=arguments-differ
        backends = self.backends

        filters = filters or {}
        for key, value in filters.items():
            backends = {name: instance for name, instance in backends.items()
                        if instance.configuration.get(key) == value}

        return list(backends.values())