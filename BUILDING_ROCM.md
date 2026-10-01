# Building Qiskit Aer with ROCm GPU Support

This guide explains how to build Qiskit Aer with AMD ROCm GPU acceleration support.

## Prerequisites

### System Requirements
- AMD GPU with ROCm support (CDNA/RDNA architecture)
- Linux operating system (Ubuntu, RHEL, SLES, or similar)
- ROCm 5.0 or later (ROCm 6.4.2 recommended for best stability, ROCm 7.0+ also supported)

### Important Notes on ROCm Versions
- **ROCm 6.4.2**: Best stability for production use (validated on MI300X)
- **ROCm 7.2+**: Latest features but may have HIP stream capture compatibility issues with 32+ qubit circuits
- If using ROCm 7.2+ and encountering `hipErrorStreamCaptureUnsupported` errors with large circuits, consider downgrading to ROCm 6.4.2

### Required Packages

#### Essential System Libraries
```bash
# OpenMP library (REQUIRED - prevents libomp.so errors)
sudo apt-get update && sudo apt-get install -y libomp-dev

# System LAPACK (required for auxiliary functions)
sudo apt install liblapack-dev

# Python and build tools
sudo apt install python3-dev python3-pip python3-venv
```

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
# Install Python build dependencies (includes pybind11)
pip install cmake conan scikit-build pybind11
```

**Note**: The `pybind11` package is required for building the C++/Python bindings. Without it, you'll get compilation errors like `fatal error: 'pybind11/pybind11.h' file not found`.

**Note**: `libomp-dev` is essential to avoid runtime errors like `ImportError: libomp.so: cannot open shared object file`. Install it before building.

## Build Instructions

### Complete Build Process (Recommended)

Follow these steps for a clean, reliable build:

```bash
# 1. Clean build artifacts (if rebuilding)
rm -rf _skbuild dist build

# 2. Install pybind11 first (critical dependency)
pip install pybind11

# 3. Install remaining Python dependencies
pip install -r requirements-dev.txt

# 4. Set environment variables
export ROCM_PATH=/opt/rocm
export AER_THRUST_BACKEND=ROCM
export QISKIT_AER_PACKAGE_NAME='qiskit-aer-gpu-rocm'

# 5. Build with explicit compiler flags
python3 setup.py bdist_wheel -- \
  -DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
  -DAER_THRUST_BACKEND=ROCM

# 6. Install the wheel
pip install dist/qiskit_aer_gpu_rocm-*.whl

# 7. Install Qiskit (not included in qiskit-aer-gpu-rocm)
pip install qiskit
```

### Step-by-Step Explanation

#### 1. Clean Build Artifacts
```bash
rm -rf _skbuild dist build
```
Removes previous build files to ensure a clean build. Essential when rebuilding or switching ROCm versions.

#### 2. Install pybind11
```bash
pip install pybind11
```
Must be installed **before** building to provide C++/Python binding headers.

#### 3. Install Build Dependencies
```bash
pip install -r requirements-dev.txt
```
Installs `cmake`, `conan`, `scikit-build`, and other required Python packages.

#### 4. Set Environment Variables
```bash
export ROCM_PATH=/opt/rocm
export AER_THRUST_BACKEND=ROCM
export QISKIT_AER_PACKAGE_NAME='qiskit-aer-gpu-rocm'
```
- `ROCM_PATH`: ROCm installation directory
- `AER_THRUST_BACKEND`: Tells the build system to use ROCm backend
- `QISKIT_AER_PACKAGE_NAME`: Names the package as `qiskit-aer-gpu-rocm`

#### 5. Build with Compiler Flags
```bash
python3 setup.py bdist_wheel -- \
  -DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
  -DAER_THRUST_BACKEND=ROCM
```
Explicitly specifies ROCm's clang++ compiler for both C++ and HIP compilation. The `--` separator passes flags to CMake.

**Note**: You may see `-DAER_THRUST_BACKEND=ROCM: command not found` at the end - this is harmless and can be ignored. The build completes successfully before this message appears.

#### 6. Install the Wheel
```bash
pip install dist/qiskit_aer_gpu_rocm-*.whl
```
Installs the compiled ROCm-enabled Qiskit Aer package.

#### 7. Install Qiskit
```bash
pip install qiskit
```
The `qiskit-aer-gpu-rocm` package doesn't include Qiskit itself, so install it separately.

### Alternative: Quick Build (Auto-Detection)

If you have a standard ROCm installation, the build system can auto-detect settings:

```bash
export AER_THRUST_BACKEND=ROCM
export QISKIT_AER_PACKAGE_NAME='qiskit-aer-gpu-rocm'
pip install -r requirements-dev.txt
python3 setup.py bdist_wheel
pip install dist/qiskit_aer_gpu_rocm-*.whl
```

However, the explicit compiler flag method is more reliable, especially in multi-ROCm environments.

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

### Runtime Error: libomp.so Not Found

**Error**: `ImportError: libomp.so: cannot open shared object file: No such file or directory`

**Solution**:
```bash
# Install OpenMP library
sudo apt-get update && sudo apt-get install -y libomp-dev

# Update library cache
sudo ldconfig

# If still not found, create symlink manually
sudo ln -sf /lib/llvm-14/lib/libomp.so /lib/x86_64-linux-gnu/libomp.so
sudo ldconfig
```

This error occurs when the OpenMP library is missing or the dynamic linker can't find it. Always install `libomp-dev` before building.

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

### Multi-GPU Issues: Runtime Errors on 32+ Qubit Circuits

**Error**: `std::runtime_error: ChunkContainer::Execute in Chunk SWAP : hipErrorStreamCaptureUnsupported`

**Cause**: ROCm 7.2+ has compatibility issues with HIP stream capture used by qiskit-aer's blocking mechanism for large circuits.

**Solutions**:
1. **Downgrade to ROCm 6.4.2** (recommended for production):
   - ROCm 6.4.2 has stable stream capture support
   - Validated on MI300X systems

2. **Set GPU visibility** (for multi-GPU systems):
   ```bash
   export ROCR_VISIBLE_DEVICES=0
   export HIP_VISIBLE_DEVICES=0
   ulimit -s unlimited
   ```

3. **Avoid 32-qubit blocking regime**:
   - Test up to 30 qubits (no blocking needed)
   - Use `complex64` precision to potentially avoid the issue

### "Command Not Found" After Build

If you see `-DAER_THRUST_BACKEND=ROCM: command not found` at the end of the build:
- **This is harmless** - the build completed successfully before this message
- The wheel file was already created in the `dist/` directory
- Simply proceed to install the wheel

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
