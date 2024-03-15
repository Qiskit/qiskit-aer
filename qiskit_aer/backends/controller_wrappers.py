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
"""
Aer controll wrappers
"""
import importlib

""" get wrapper with suffix """
def try_import_backend(backend_module_suffix):
    module_name = f".controller_wrappers_{backend_module_suffix}"
    try:
        return importlib.import_module(module_name, "qiskit_aer.backends")
    except ImportError:
        return None


IMPORTED_BACKEND = None
BACKENDS = ["cuda", "rocm", "cpu"]

for backend_suffix in BACKENDS:
    backend_module = try_import_backend(backend_suffix)
    if backend_module:
        IMPORTED_BACKEND = backend_suffix
        globals().update(
            {
                name: getattr(backend_module, name)
                for name in dir(backend_module)
                if not name.startswith("_")
            }
        )
        break

if IMPORTED_BACKEND is None:
    raise ImportError("No backend found for qiskit-aer.")
