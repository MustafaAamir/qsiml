import time
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from qsiml import QuantumCircuit

def benchmark_circuit(n_qubits, depth):
    print(f"Benchmarking {n_qubits} qubits, depth {depth}...")
    qc = QuantumCircuit(n_qubits)
    
    start_time = time.time()
    for _ in range(depth):
        for i in range(n_qubits):
            qc.h(i)
        for i in range(n_qubits - 1):
            qc.cnot(i, i+1)
            

    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    return end_time - start_time

if __name__ == "__main__":
    benchmark_circuit(10, 5)
    benchmark_circuit(12, 5)
    benchmark_circuit(15, 5)
    benchmark_circuit(18, 5)
    # 20 might take a bit longer but should be fine (2^20 * 16 bytes ~ 16MB state vector)
    benchmark_circuit(20, 5)
