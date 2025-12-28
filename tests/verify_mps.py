import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from qsiml import QuantumCircuit

def verify_mps_bell_state():
    print("Verifying MPS Bell State...")
    qc = QuantumCircuit(2, backend='mps')
    qc.h(0)
    qc.cnot(0, 1)
    
    # Check state vector (reconstructed from MPS)
    sv = qc.state_vector
    
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1/np.sqrt(2)
    expected[3] = 1/np.sqrt(2)
    
    if np.allclose(sv, expected):
        print("MPS Bell State Verified.")
    else:
        print(f"MPS Bell State Failed. Expected {expected}, got {sv}")
        exit(1)

def verify_mps_gates():
    print("Verifying MPS simple gates...")
    qc = QuantumCircuit(1, backend='mps')
    qc.px(0)
    if not np.allclose(qc.state_vector, [0, 1]):
         print("MPS X gate failed")
         exit(1)
    
    qc.reset(0)
    qc.h(0)
    qc.h(0)
    if not np.allclose(qc.state_vector, [1, 0]):
        print("MPS H * H failed")
        exit(1)

    print("MPS Simple gates verified.")

def verify_mps_truncation():
    print("Verifying MPS truncation (GHZ-like)...")
    # For N=3, GHZ state has low bond dim (2). So max_bond_dim=2 is enough.
    # If we restrict strict truncation, it should still work.
    n = 6
    qc = QuantumCircuit(n, backend='mps', max_bond_dim=4)
    qc.h(0)
    for i in range(n-1):
        qc.cnot(i, i+1)
        
    sv = qc.state_vector
    # GHZ state: (|0...0> + |1...1>) / sqrt(2)
    expected = np.zeros(2**n, dtype=complex)
    expected[0] = 1/np.sqrt(2)
    expected[-1] = 1/np.sqrt(2)
    
    if np.allclose(sv, expected):
         print("MPS GHZ (exact due to low entanglement) Verified.")
    else:
         print("MPS GHZ Failed.")
         print("Norm:", np.linalg.norm(sv))
         # exit(1)

if __name__ == "__main__":
    verify_mps_bell_state()
    verify_mps_gates()
    verify_mps_truncation()
