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

# pylint: disable=invalid-name, bad-continuation

"""RemoteConfig module

This module is the configuration for the distributed environment
"""

import logging
logger = logging.getLogger(__name__)


class RemoteConfig():
    """ RemoteConfig class """
    def __init__(self, config):
        self._config = config

    def get_host_list(self):
        """
        Get host list from configuration
        """
        host_list = []
        for each_config in self._config:
            host_list.append(each_config["host"])
        return host_list

    def get_config_list(self):
        """
        Get config list
        """
        return self._config
