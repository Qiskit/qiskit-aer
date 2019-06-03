# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Provider for AerCLoud"""

from qiskit.providers import BaseProvider
from .aercloudbackend import AerCloudBackend

class AerCloudProvider(BaseProvider):
    def __init__(self, **kwargs):
        super().__init__()

    def backends(self, name=None, filters=None, **kwargs):
        # pylint: disable=arguments-differ

        _bk = AerCloudBackend(provider=self, kwargs=kwargs)
        if name == _bk.name():
            self._backends = [_bk]
        else:
            self._backends = []

        return self._backends
    
    def get_backend(self, name=None, **kwargs):
        return super().get_backend(name=name, **kwargs)
