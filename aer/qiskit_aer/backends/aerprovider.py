# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name, bad-continuation

"""Provider for Qiskit Aer backends."""

from qiskit.backends import BaseProvider
from qiskit.backends.providerutils import filter_backends

from .qasm_simulator import QasmSimulator
from .statevector_simulator import StatevectorSimulator


class AerProvider(BaseProvider):
    """Provider for Qiskit Aer backends."""

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

        # Populate the list of local Sympy backends.
        self._backends = [QasmSimulator(provider=self),
                          StatevectorSimulator(provider=self)]

    def get_backend(self, name=None, **kwargs):
        return super().get_backend(name=name, **kwargs)

    def backends(self, name=None, filters=None, **kwargs):
        # pylint: disable=arguments-differ
        if name:
            kwargs.update({'name': name})

        return filter_backends(self._backends, filters=filters, **kwargs)

    def __str__(self):
        return 'AerProvider'
