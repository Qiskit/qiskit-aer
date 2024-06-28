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


from qiskit.providers import QiskitBackendNotFoundError
from qiskit.providers.providerutils import filter_backends

from .backends.aer_simulator import AerSimulator
from .backends.qasm_simulator import QasmSimulator
from .backends.statevector_simulator import StatevectorSimulator
from .backends.unitary_simulator import UnitarySimulator


class AerProvider:
    """Provider for Aer backends."""

    _BACKENDS = None
    version = 1

    @staticmethod
    def _get_backends():
        if AerProvider._BACKENDS is None:
            # Populate the list of Aer simulator backends.
            methods = AerSimulator().available_methods()
            devices = AerSimulator().available_devices()
            backends = []
            for method in methods:
                for device in devices:
                    name = "aer_simulator"
                    if method in [None, "automatic"]:
                        backends.append((name, AerSimulator, method, device))
                    else:
                        name += f"_{method}"
                        if method == "tensor_network":
                            if device == "GPU":
                                name += f"_{device}".lower()
                                backends.append((name, AerSimulator, method, device))
                        else:
                            if device == "CPU":
                                backends.append((name, AerSimulator, method, device))
                            elif method in ["statevector", "density_matrix", "unitary"]:
                                name += f"_{device}".lower()
                                backends.append((name, AerSimulator, method, device))

            # Add legacy backend names
            backends += [
                ("qasm_simulator", QasmSimulator, None, None),
                ("statevector_simulator", StatevectorSimulator, None, None),
                ("unitary_simulator", UnitarySimulator, None, None),
            ]
            AerProvider._BACKENDS = backends

        return AerProvider._BACKENDS

    def get_backend(self, name=None, **kwargs):
        """Return a single Aer backend matching the specified filtering.

        Args:
            name (str): name of the Aer backend.
            **kwargs: dict used for filtering.

        Returns:
            Backend: an Aer backend matching the filtering.

        Raises:
            QiskitBackendNotFoundError: if no backend could be found or
                more than one backend matches the filtering criteria.
        """
        backends = self.backends(name, **kwargs)
        if len(backends) > 1:
            raise QiskitBackendNotFoundError("More than one backend matches the criteria")
        if not backends:
            raise QiskitBackendNotFoundError("No backend matches the criteria")

        return backends[0]

    def backends(self, name=None, filters=None, **kwargs):
        """Return a list of backends matching the specified filtering.

        Args:
            name (str): name of the backend.
            filters (callable): filtering conditions as a callable.
            **kwargs: dict used for filtering.

        Returns:
            list[Backend]: a list of Backends that match the filtering
                criteria.
        """
        # pylint: disable=unused-argument
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
