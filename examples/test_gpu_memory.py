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
            
            lines = result.stdout.split('\n')
            for line in lines:
                if 'VRAM Total Memory' in line or 'Total' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.replace('.', '').isdigit():
                            mem_mb = float(part)
                            if i + 1 < len(parts):
                                unit = parts[i + 1].lower()
                                if 'gb' in unit:
                                    return mem_mb
                                elif 'mb' in unit:
                                    return mem_mb / 1024
                            return mem_mb / 1024 if mem_mb > 1000 else mem_mb
    except Exception as e:
        print(f"Error: {e}")
    
    return None


def calculate_max_qubits(gpu_memory_gb):
    """Calculate maximum qubits based on GPU memory."""
    if gpu_memory_gb is None:
        return 28
    
    usable_memory_bytes = gpu_memory_gb * 0.8 * (1024 ** 3)
    max_qubits = int(np.log2(usable_memory_bytes / 16))
    return min(max(max_qubits, 25), 45)


def print_memory_table(max_qubits):
    """Print memory requirements."""
    print("Memory Requirements:")
    print("-" * 60)
    print(f"{'Qubits':<10} {'Memory Required':<20} {'Fits in GPU'}")
    print("-" * 60)
    
    gpu_mem = get_gpu_memory()
    gpu_bytes = gpu_mem * (1024 ** 3) if gpu_mem else 0
    
    for q in range(10, min(max_qubits + 5, 46), 5):
        mem_bytes = (2 ** q) * 16
        mem_gb = mem_bytes / (1024 ** 3)
        mem_tb = mem_bytes / (1024 ** 4)
        
        if mem_tb >= 1:
            mem_str = f"{mem_tb:.2f} TB"
        else:
            mem_str = f"{mem_gb:.2f} GB"
        
        fits = "✓" if mem_bytes < gpu_bytes * 0.8 else "✗"
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
        print(f"  (Using 80% of GPU memory as safe limit)")
        print()
        
        print_memory_table(max_qubits)
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
