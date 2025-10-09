# Building Qiskit Aer with ROCm GPU Support

This guide explains how to build Qiskit Aer with AMD ROCm GPU acceleration support.

## Prerequisites

### System Requirements
- AMD GPU with ROCm support (CDNA/RDNA architecture)
- Linux operating system (Ubuntu, RHEL, SLES, or similar)
- ROCm 5.0 or later (ROCm 7.0+ recommended)

### Required Packages

#### ROCm Libraries
```bash
# Install ROCm base packages
sudo apt install rocm-hip-runtime rocm-hip-sdk

# Install math libraries
sudo apt install rocblas rocsolver hipblas

# Install development tools
sudo apt install hip-dev rocm-cmake
```

#### Build Dependencies
```bash
# System LAPACK (required for auxiliary functions)
sudo apt install liblapack-dev

# Python and build tools
sudo apt install python3-dev python3-pip python3-venv

# Install Python build dependencies
pip install cmake conan scikit-build pybind11
```

**Note**: The `pybind11` package is required for building the C++/Python bindings. Without it, you'll get compilation errors like `fatal error: 'pybind11/pybind11.h' file not found`.

## Build Instructions

### 1. Install Python Dependencies

```bash
# Install all build dependencies including pybind11
pip install -r requirements-dev.txt
```

This will install all necessary Python packages including `pybind11`, `cmake`, `conan`, and `scikit-build`.

### 2. Set Environment Variables

```bash
export ROCM_PATH=/opt/rocm
export CMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++
export CMAKE_HIP_COMPILER=${ROCM_PATH}/llvm/bin/clang++
```

### 3. Configure Build

The build system automatically detects ROCm when `AER_THRUST_BACKEND=ROCM` is set:

```bash
export AER_THRUST_BACKEND=ROCM
export QISKIT_AER_PACKAGE_NAME='qiskit-aer-gpu-rocm'
```

### 4. Build

```bash
python3 setup.py bdist_wheel
```

Or use pip:

```bash
pip install . -v
```

### 5. Install

```bash
pip install dist/qiskit_aer_gpu_rocm-*.whl
```

## Architecture Details

### Library Dependencies

The ROCm build links against three key libraries:

1. **rocBLAS** (`librocblas.so`) - GPU-accelerated BLAS operations
   - Matrix multiplications, vector operations
   - Optimized for AMD GPUs

2. **rocSOLVER** (`librocsolver.so`) - GPU-accelerated LAPACK routines
   - Linear system solvers (dgesv, dgetrf, etc.)
   - Matrix factorizations
   - Eigenvalue computations

3. **System LAPACK** (`liblapack.so.3`) - CPU-based auxiliary functions
   - Machine precision routines (dlamch_)
   - Utility functions not provided by rocSOLVER
   - Required for complete LAPACK compatibility

### Why All Three Libraries?

- **rocBLAS + rocSOLVER** alone don't provide all LAPACK functions
- **rocSOLVER** focuses on high-level linear algebra solvers
- **System LAPACK** provides low-level auxiliary routines
- The combination ensures full functionality while maximizing GPU acceleration

## Verification

Test the installation:

```bash
cd /tmp  # Run outside source directory to avoid import issues
python3 -c '
from qiskit_aer import AerSimulator
sim = AerSimulator(device="GPU")
print("Available devices:", sim.available_devices())
'
```

Expected output:
```
Available devices: ('CPU', 'GPU')
```

### Performance Testing

Run the included benchmark scripts to verify GPU acceleration:

```bash
# Quick test (5-10 seconds)
cd /tmp && python3 ~/playground/qiskit-aer/examples/quick_gpu_test.py

# Comprehensive benchmark (2-3 minutes)
cd /tmp && python3 ~/playground/qiskit-aer/examples/rocm_gpu_benchmark.py
```

See [examples/README.md](examples/README.md) for more details and custom examples.

## Troubleshooting

### Build Errors: pybind11 Not Found

If you see `fatal error: 'pybind11/pybind11.h' file not found`:
```bash
pip install pybind11
# Or install all dependencies:
pip install -r requirements-dev.txt
```

### Symbol Not Found Errors

If you see `undefined symbol: dlamch_` or similar errors:
- Ensure system LAPACK is installed: `sudo apt install liblapack-dev`
- Rebuild with clean build directory: `rm -rf _skbuild dist build`

### GPU Not Detected

Check ROCm installation:
```bash
rocminfo
hipconfig
```

Verify GPU is visible:
```bash
/opt/rocm/bin/rocm-smi
```

### Import Errors When Testing

Always test from outside the source directory:
```bash
cd /tmp
python3 -c 'from qiskit_aer import AerSimulator'
```

If you test from the source directory, Python may try to import from the source tree instead of the installed package.

## Performance Notes

- The ROCm backend uses HIP for GPU kernels
- Wavefront size is set to 64 for CDNA architectures (MI100, MI200, MI300)
- Shared memory limits are architecture-specific
- For optimal performance, ensure your GPU has sufficient memory for the quantum circuit

## Build Warnings (Non-Critical)

During the build, you may see several deprecation warnings. These are **non-critical** and don't affect functionality:

### CMake Warnings
- **CMP0148 Policy Warning**: Uses older FindPythonLibs (will be updated to FindPython3 in future)
- Safe to ignore with `-Wno-dev` flag if desired

### C++ Deprecation Warnings
- **pybind11 `get_type()`**: Will be updated to `py::type::of()` in future releases
- **Thrust `unary_function`**: Legacy Thrust API (ROCm 7.x includes updated Thrust, will be modernized)
- **NaN/infinity checks**: Due to compiler optimization flags, safe to ignore

These warnings don't impact the built binary's functionality or performance.

## Known Limitations

1. ROCm support requires Linux
2. Some legacy AMD GPUs may not be supported
3. Double precision performance varies by GPU architecture
4. Memory requirements scale with number of qubits

## Additional Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Qiskit Aer Documentation](https://qiskit.org/ecosystem/aer/)
- [AMD GPU Support Matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

## Contributing

If you encounter issues with ROCm builds, please report them with:
- ROCm version (`/opt/rocm/bin/hipconfig --version`)
- GPU model (`rocminfo | grep "Name:"`)
- Build log output
- Error messages

## License

This software is licensed under the Apache License 2.0. See LICENSE.txt for details.
