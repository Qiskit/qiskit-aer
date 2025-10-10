#!/usr/bin/env python3
"""
Quick test for GPU memory detection and max qubits calculation.
"""

import subprocess
import numpy as np

def get_gpu_memory():
    """Get GPU memory in GB using rocm-smi."""
    try:
        result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("ROCm SMI Output:")
            print(result.stdout)
            print()
            
            # Parse output to find VRAM size in bytes
            # Format: "GPU[0]          : VRAM Total Memory (B): 206141652992"
            lines = result.stdout.split('\n')
            for line in lines:
                if 'VRAM Total Memory (B)' in line and 'GPU[0]' in line:
                    # Extract the bytes value after the last colon
                    parts = line.split(':')
                    if len(parts) >= 3:
                        bytes_str = parts[-1].strip()
                        try:
                            memory_bytes = float(bytes_str)
                            # Convert bytes to GB
                            memory_gb = memory_bytes / (1024 ** 3)
                            return memory_gb
                        except ValueError:
                            continue
    except Exception as e:
        print(f"Error: {e}")
    
    return None


def calculate_max_qubits(gpu_memory_gb):
    """Calculate maximum qubits based on GPU memory."""
    if gpu_memory_gb is None:
        return 28
    
    # Use 70% of GPU memory for safety (accounting for overhead)
    usable_memory_bytes = gpu_memory_gb * 0.7 * (1024 ** 3)
    max_qubits = int(np.log2(usable_memory_bytes / 16))
    return min(max(max_qubits, 25), 45)


def print_memory_table(max_qubits, gpu_memory_gb):
    """Print memory requirements."""
    print("Memory Requirements:")
    print("-" * 60)
    print(f"{'Qubits':<10} {'Memory Required':<20} {'Fits in GPU'}")
    print("-" * 60)
    
    gpu_bytes = gpu_memory_gb * (1024 ** 3) if gpu_memory_gb else 0
    
    for q in range(10, min(max_qubits + 5, 46), 5):
        mem_bytes = (2 ** q) * 16
        mem_gb = mem_bytes / (1024 ** 3)
        mem_tb = mem_bytes / (1024 ** 4)
        
        if mem_tb >= 1:
            mem_str = f"{mem_tb:.2f} TB"
        else:
            mem_str = f"{mem_gb:.2f} GB"
        
        # Use 70% threshold for safety
        fits = "✓" if mem_bytes < gpu_bytes * 0.7 else "✗"
        print(f"{q:<10} {mem_str:<20} {fits}")
    
    print("-" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("GPU Memory Detection Test")
    print("=" * 60)
    print()
    
    gpu_memory = get_gpu_memory()
    
    if gpu_memory:
        print(f"✓ GPU Memory Detected: {gpu_memory:.1f} GB")
        print()
        
        max_qubits = calculate_max_qubits(gpu_memory)
        print(f"✓ Estimated Max Qubits: {max_qubits}")
        print(f"  (Using 70% of GPU memory as safe limit)")
        print()
        
        print_memory_table(max_qubits, gpu_memory)
        print()
        
        print("Recommended benchmark range:")
        if max_qubits >= 40:
            print("  [10, 15, 20, 25, 28, 30, 32, 35, 38, 40]")
        elif max_qubits >= 35:
            print("  [10, 15, 20, 25, 28, 30, 32, 35]")
        elif max_qubits >= 30:
            print("  [10, 15, 20, 25, 28, 30]")
        else:
            print("  [10, 15, 20, 25]")
        
    else:
        print("✗ Could not detect GPU memory")
        print("  Using conservative default: 28 qubits")
