import importlib


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
