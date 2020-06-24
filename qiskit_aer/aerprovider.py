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

"""Provider for Qiskit Aer backends."""

from qiskit.providers import BaseProvider
from qiskit.providers.providerutils import filter_backends

from .backends.qasm_simulator import QasmSimulator
from .backends.statevector_simulator import StatevectorSimulator
from .backends.unitary_simulator import UnitarySimulator
from .backends.pulse_simulator import PulseSimulator


class AerProvider(BaseProvider):
    """Provider for Qiskit Aer backends."""

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

        # Populate the list of Aer simulator providers.
        self._backends = [QasmSimulator(provider=self),
                          StatevectorSimulator(provider=self),
                          UnitarySimulator(provider=self),
                          PulseSimulator(provider=self)]

    def get_backend(self, name=None, **kwargs):
        return super().get_backend(name=name, **kwargs)

    def backends(self, name=None, filters=None, **kwargs):
        # pylint: disable=arguments-differ
        backends = self._backends
        if name:
            backends = [backend for backend in backends if backend.name() == name]

        return filter_backends(backends, filters=filters, **kwargs)

    def __str__(self):
        return 'AerProvider'
