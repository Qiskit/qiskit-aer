#!/usr/bin/env bash
#
# ROCm GPU Detection and Configuration Helper
# Usage: ./tools/detect_rocm.sh
#
# This script helps users detect their AMD GPU and generate
# the correct build configuration for Qiskit Aer with ROCm support.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Qiskit Aer - ROCm GPU Detection & Configuration      ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

check_rocm() {
    echo -e "${YELLOW}[1/4] Checking ROCm Installation...${NC}"
    
    if ! command -v rocminfo &> /dev/null; then
        echo -e "${RED}✗ ROCm not found!${NC}"
        echo "  Install ROCm from: https://rocm.docs.amd.com/"
        exit 1
    fi
    
    if [ -f /opt/rocm/.info/version ]; then
        ROCM_VERSION=$(cat /opt/rocm/.info/version)
    else
        ROCM_VERSION="unknown"
    fi
    
    echo -e "${GREEN}✓ ROCm installed: version ${ROCM_VERSION}${NC}"
    
    if [[ "$ROCM_VERSION" < "5.0" ]] && [[ "$ROCM_VERSION" != "unknown" ]]; then
        echo -e "${YELLOW}  Warning: ROCm 5.0+ recommended (you have ${ROCM_VERSION})${NC}"
    elif [[ "$ROCM_VERSION" > "7.0" ]] || [[ "$ROCM_VERSION" == "7."* ]]; then
        echo -e "${GREEN}  Latest ROCm 7.x - Excellent!${NC}"
    fi
    echo ""
}

detect_gpus() {
    echo -e "${YELLOW}[2/4] Detecting AMD GPUs...${NC}"
    
    GPU_LIST=$(rocminfo | grep "Name:" | grep "gfx" || true)
    
    if [ -z "$GPU_LIST" ]; then
        echo -e "${RED}✗ No AMD GPUs detected!${NC}"
        echo "  Make sure:"
        echo "  - Your GPU is properly installed"
        echo "  - You have proper drivers/kernel modules"
        echo "  - Run: ls /dev/dri/"
        exit 1
    fi
    
    # Extract unique architectures (filter to simple gfx format only)
    # This filters out verbose names like "amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"
    GPU_ARCHS=$(echo "$GPU_LIST" | awk '{print $2}' | grep -oE '^gfx[0-9a-f]+$' | sort -u)
    
    echo -e "${GREEN}✓ Found AMD GPU(s):${NC}"
    for arch in $GPU_ARCHS; do
        case $arch in
            gfx908)
                echo "  • ${arch} - AMD MI100 (32GB HBM2)"
                MEMORY_REC="25"
                ;;
            gfx90a)
                echo "  • ${arch} - AMD MI210/MI250/MI250X (64-128GB HBM2e)"
                MEMORY_REC="27"
                ;;
            gfx940|gfx941|gfx942)
                echo "  • ${arch} - AMD MI300 series (192GB HBM3)"
                MEMORY_REC="28"
                ;;
            gfx1030|gfx1031|gfx1032)
                echo "  • ${arch} - AMD RX 6000 series (RDNA 2)"
                MEMORY_REC="24"
                echo "    Note: Consumer GPU, limited HPC features"
                ;;
            gfx1100|gfx1101|gfx1102|gfx1103)
                echo "  • ${arch} - AMD RX 7000 series (RDNA 3)"
                MEMORY_REC="25"
                echo "    Note: Consumer GPU, limited HPC features"
                ;;
            *)
                echo "  • ${arch} - Unknown/Untested architecture"
                MEMORY_REC="23"
                ;;
        esac
    done
    echo ""
}

generate_build_cmd() {
    echo -e "${YELLOW}[3/4] Generating Build Configuration...${NC}"
    
    ARCH_LIST=$(echo "$GPU_ARCHS" | tr '\n' ' ' | xargs)
    
    cat > /tmp/qiskit_aer_rocm_build.sh <<'EOF'
#!/bin/bash
# Generated ROCm build script for Qiskit Aer
# GPU Architectures: ARCH_LIST_PLACEHOLDER
# Generated: DATE_PLACEHOLDER

set -e

# Set environment variables
export ROCM_PATH=/opt/rocm
export AER_THRUST_BACKEND=ROCM
export AER_ROCM_ARCH="ARCH_LIST_PLACEHOLDER"
export CMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++
export CMAKE_HIP_COMPILER=${ROCM_PATH}/llvm/bin/clang++

# Ensure we're in the qiskit-aer directory
if [ ! -f "setup.py" ] || [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please run this script from the qiskit-aer root directory"
    exit 1
fi

# Detect Clang version and configure Conan if needed
CLANG_VERSION=$($CMAKE_CXX_COMPILER --version | grep -oP 'clang version \K[0-9]+' | head -1)
echo "Detected Clang version: $CLANG_VERSION"

# Workaround for Conan 1.x with Clang > 17 (ROCm 7.0+)
if [ "$CLANG_VERSION" -gt 17 ] 2>/dev/null; then
    echo "Note: Clang $CLANG_VERSION detected. Configuring Conan for compatibility..."
    
    # Update Conan profile to use Clang 17 (closest supported version)
    if [ ! -f ~/.conan/profiles/default ]; then
        conan profile new default --detect 2>/dev/null || true
    fi
    
    # Modify the Conan profile to use Clang 17
    conan profile update settings.compiler.version=17 default 2>/dev/null || true
    
    echo "Conan profile updated to use Clang 17 for compatibility"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
python3 -m pip install -r requirements-dev.txt

# Build wheel
echo "Building qiskit-aer-gpu-rocm wheel..."
QISKIT_AER_PACKAGE_NAME='qiskit-aer-gpu-rocm' \
    python3 setup.py bdist_wheel -- \
        -DAER_THRUST_BACKEND=ROCM \
        -DAER_ROCM_ARCH="ARCH_LIST_PLACEHOLDER" \
        -DCMAKE_BUILD_TYPE=Release

# Install
echo "Installing wheel..."
python3 -m pip install --force-reinstall dist/qiskit_aer_gpu_rocm-*.whl

echo ""
echo "✓ Build complete!"
echo ""
echo "Test with:"
echo "  python3 -c 'from qiskit_aer import AerSimulator; sim = AerSimulator(device=\"GPU\"); print(\"GPU available:\", sim.available_devices())'"
EOF

    # Replace placeholders
    sed -i "s/ARCH_LIST_PLACEHOLDER/${ARCH_LIST}/g" /tmp/qiskit_aer_rocm_build.sh
    sed -i "s/DATE_PLACEHOLDER/$(date)/g" /tmp/qiskit_aer_rocm_build.sh

    chmod +x /tmp/qiskit_aer_rocm_build.sh
    
    echo -e "${GREEN}✓ Build script generated: /tmp/qiskit_aer_rocm_build.sh${NC}"
    echo ""
}

show_recommendations() {
    echo -e "${YELLOW}[4/4] Performance Recommendations...${NC}"
    echo ""
    echo "Memory Settings (blocking_qubits):"
    for arch in $GPU_ARCHS; do
        case $arch in
            gfx908) echo "  ${arch}: blocking_qubits=25 (MI100, 32GB)" ;;
            gfx90a) echo "  ${arch}: blocking_qubits=27 (MI200, 64-128GB)" ;;
            gfx940|gfx941|gfx942) echo "  ${arch}: blocking_qubits=28 (MI300, 192GB)" ;;
            gfx1030|gfx1031|gfx1032) echo "  ${arch}: blocking_qubits=24 (RX 6000)" ;;
            gfx1100|gfx1101|gfx1102) echo "  ${arch}: blocking_qubits=25 (RX 7000)" ;;
            *) echo "  ${arch}: blocking_qubits=23 (conservative)" ;;
        esac
    done
    echo ""
    echo "Example usage in Python:"
    cat <<'EOF'
    from qiskit_aer import AerSimulator
    
    sim = AerSimulator(method='statevector', device='GPU')
    result = sim.run(circuit, 
                     blocking_enable=True,
                     blocking_qubits=27,  # Adjust for your GPU
                     shots=1000).result()
EOF
    echo ""
}

show_next_steps() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                    Next Steps                          ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "1. Review the generated build script:"
    echo -e "   ${GREEN}cat /tmp/qiskit_aer_rocm_build.sh${NC}"
    echo ""
    echo "2. Run the build from qiskit-aer directory:"
    echo -e "   ${GREEN}bash /tmp/qiskit_aer_rocm_build.sh${NC}"
    echo ""
    echo "3. Or copy environment variables:"
    echo -e "   ${GREEN}export ROCM_PATH=/opt/rocm${NC}"
    echo -e "   ${GREEN}export AER_THRUST_BACKEND=ROCM${NC}"
    echo -e "   ${GREEN}export AER_ROCM_ARCH=\"${ARCH_LIST}\"${NC}"
    echo ""
    echo "4. Quick test after installation:"
    echo -e "   ${GREEN}python3 -c 'from qiskit_aer import AerSimulator; sim = AerSimulator(device=\"GPU\"); print(sim.available_devices())'${NC}"
    echo ""
}

# Main execution
print_header
check_rocm
detect_gpus
generate_build_cmd
show_recommendations
show_next_steps

echo -e "${GREEN}Done!${NC}"
