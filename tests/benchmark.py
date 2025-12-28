import time
import numpy as np
import sys
import os

# Add src to path

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from qsiml_og import QuantumCircuit

def benchmark_circuit(n_qubits, depth, og):
    if og: 
        from qsiml_og import QuantumCircuit
    else:
        from qsiml import QuantumCircuit

    print(f"Benchmarking {n_qubits} qubits, depth {depth}...")
    qc = QuantumCircuit(n_qubits)
    
    start_time = time.time()
    for _ in range(depth):
        for i in range(n_qubits):
            qc.h(i)
        for i in range(n_qubits - 1):
            qc.cnot(i, i+1)

    # qc.dump(msg="Before measure") 
    # For comparison, we want the state BEFORE measurement collapse if possible, 
    # but measure_all collapses it.
    # To compare 'results', let's compare the state vector before measurement.
    
    if og:
         qc._eval_state_vector()
    
    final_state = qc.state_vector.copy()

    # qc.measure_all() # Skipping measure to keep state for comparison

    end_time = time.time()
    # print(qc._katas())
    
    return end_time - start_time, final_state

if __name__ == "__main__":
    # Validate correctness on a smaller scale first to fail fast if wrong
    print("Validating correctness on N=10...")
    t1, sv1 = benchmark_circuit(10, 5, False)
    t2, sv2 = benchmark_circuit(10, 5, True)
    
    if np.allclose(sv1, sv2):
        print("SUCCESS: State vectors match for N=10.")
    else:
        print("FAILURE: State vectors mismatch for N=10!")
        print("Max diff:", np.max(np.abs(sv1 - sv2)))
        exit(1)
        
    print("\nRunning Benchmark N=20...")
    t_opt, _ = benchmark_circuit(20, 5, False)
    print(f"Optimized Time: {t_opt:.4f} seconds")
    
    t_og, _ = benchmark_circuit(20, 5, True)
    print(f"OG Time: {t_og:.4f} seconds")
    
    print(f"Speedup: {t_og / t_opt:.2f}x")
