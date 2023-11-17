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

# pylint: disable=invalid-name
"""Provider for Aer backends."""


from qiskit.providers import ProviderV1 as Provider
from qiskit.providers.providerutils import filter_backends

from .backends.aer_simulator import AerSimulator
from .backends.qasm_simulator import QasmSimulator
from .backends.statevector_simulator import StatevectorSimulator
from .backends.unitary_simulator import UnitarySimulator


class AerProvider(Provider):
    """Provider for Aer backends."""

    _BACKENDS = None

    @staticmethod
    def _get_backends():
        if AerProvider._BACKENDS is None:
            # Populate the list of Aer simulator backends.
            methods = AerSimulator().available_methods()
            devices = AerSimulator().available_devices()
            backends = []
            for method in methods:
                name = "aer_simulator"
                if method not in [None, "automatic"]:
                    name += f"_{method}"
                device_name = "CPU"
                backends.append((name, AerSimulator, method, device_name))

                # Add GPU device backends
                if method in ["statevector", "density_matrix", "unitary"]:
                    for device in devices:
                        if device != "CPU":
                            new_name = f"{name}_{device}".lower()
                            device_name = device
                            backends.append((new_name, AerSimulator, method, device_name))

            # Add legacy backend names
            backends += [
                ("qasm_simulator", QasmSimulator, None, None),
                ("statevector_simulator", StatevectorSimulator, None, None),
                ("unitary_simulator", UnitarySimulator, None, None),
            ]
            AerProvider._BACKENDS = backends

        return AerProvider._BACKENDS

    def get_backend(self, name=None, **kwargs):
        return super().get_backend(name=name, **kwargs)

    def backends(self, name=None, filters=None, **kwargs):
        # pylint: disable=arguments-differ
        # Instantiate a new backend instance so if config options
        # are set they will only last as long as that backend object exists
        backends = []

        # pylint: disable=not-an-iterable
        # pylint infers _get_backends to always return None
        for backend_name, backend_cls, method, device in self._get_backends():
            opts = {"provider": self}
            if method is not None:
                opts["method"] = method
            if device is not None:
                opts["device"] = device
            if name is None or backend_name == name:
                backends.append(backend_cls(**opts))
        return filter_backends(backends, filters=filters)

    def __str__(self):
        return "AerProvider"
