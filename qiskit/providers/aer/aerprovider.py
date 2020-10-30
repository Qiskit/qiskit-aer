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

        # Populate the list of Aer simulator backends.
        self._backends = [
            ('qasm_simulator', QasmSimulator),
            ('statevector_simulator', StatevectorSimulator),
            ('unitary_simulator', UnitarySimulator),
            ('pulse_simulator', PulseSimulator)
        ]

    def get_backend(self, name=None, **kwargs):
        return super().get_backend(name=name, **kwargs)

    def backends(self, name=None, filters=None, **kwargs):
        # pylint: disable=arguments-differ
        # Instantiate a new backend instance so if config options
        # are set they will only last as long as that backend object exists
        backends = []
        for backend_name, backend_cls in self._backends:
            if name is None or backend_name == name:
                backends.append(backend_cls(provider=self))
        return filter_backends(backends, filters=filters)

    def __str__(self):
        return 'AerProvider'
