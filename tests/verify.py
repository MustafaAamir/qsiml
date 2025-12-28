import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from qsiml import QuantumCircuit

def verify_bell_state():
    print("Verifying Bell State...")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cnot(0, 1)

    sv = qc.state_vector
    
    # Expected: 1/sqrt(2) (|00> + |11>)
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1/np.sqrt(2)
    expected[3] = 1/np.sqrt(2)
    
    if np.allclose(sv, expected):
        print("Bell State Verified.")
    else:
        print(f"Bell State Failed. Expected {expected}, got {sv}")
        exit(1)

def verify_gates():
    print("Verifying simple gates...")
    qc = QuantumCircuit(1)
    qc.px(0)

    if not np.allclose(qc.state_vector, [0, 1]):
         print("X gate failed")
         exit(1)
    
    qc.reset(0)
    qc.h(0)
    qc.h(0)

    if not np.allclose(qc.state_vector, [1, 0]):
        print("H * H failed")
        exit(1)

    print("Simple gates verified.")

if __name__ == "__main__":
    # Note: 'x' is not exposed directly in the original code, it was 'px'.
    # I should check what methods are available. The original likely used 'px'
    # but my plan mentions 'x'. I will stick to what is available or update names.
    # The original had 'px' but I will check availability.
    # Actually, let's just use 'px' for now until I refactor.
    verify_bell_state()
    verify_gates()
