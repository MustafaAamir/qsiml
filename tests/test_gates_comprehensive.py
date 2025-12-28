import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from qsiml import QuantumCircuit

class TestQuantumCircuit(unittest.TestCase):
    def setUp(self):
        np.set_printoptions(linewidth=200)

    def assertStateAlmostEqual(self, qc, expected):
        # Allow for global phase differences if needed, but for now strict check
        # Check tolerance
        self.assertTrue(np.allclose(qc.state_vector, expected, atol=1e-10), 
                        f"State mismatch.\nExpected: {expected}\nGot: {qc.state_vector}")

    def test_x_gate(self):
        qc = QuantumCircuit(2)
        qc.px(0)
        # q0 is LSB. |00> -> |01> (index 1) if q0 is rightmost?
        # Wait, let's check convention. 
        # In my implementation:
        # q0 is associated with the last axis (n-1). 
        # If flattened, index is sum(bit_i * 2^i).
        # So q0=1 means index 1. q1=1 means index 2.
        
        expected = np.zeros(4, dtype=complex)
        expected[1] = 1 # |01> (q1=0, q0=1)
        self.assertStateAlmostEqual(qc, expected)
        
        qc.px(1)
        expected = np.zeros(4, dtype=complex)
        expected[3] = 1 # |11>
        self.assertStateAlmostEqual(qc, expected)

    def test_h_gate(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        expected = np.array([1, 1]) / np.sqrt(2)
        self.assertStateAlmostEqual(qc, expected)
        
        qc.h(0)
        expected = np.array([1, 0])
        self.assertStateAlmostEqual(qc, expected)

    def test_y_gate(self):
        qc = QuantumCircuit(1)
        qc.py(0)
        # Y |0> = i|1>
        expected = np.array([0, 1j])
        self.assertStateAlmostEqual(qc, expected)
        
        qc.py(0)
        # Y(i|1>) = i(-i|0>) = |0>
        expected = np.array([1, 0])
        self.assertStateAlmostEqual(qc, expected)

    def test_z_gate(self):
        qc = QuantumCircuit(1)
        qc.px(0) # |1>
        qc.pz(0) # -|1>
        expected = np.array([0, -1])
        self.assertStateAlmostEqual(qc, expected)

    def test_cnot(self):
        # Test CNOT(control=0, target=1)
        # |00> -> |00>
        # |01> -> |11> (since q0 is control)
        qc = QuantumCircuit(2)
        qc.px(0) # |01>
        qc.cnot(0, 1) # Control q0=1, flip q1
        
        expected = np.zeros(4, dtype=complex)
        expected[3] = 1 # |11>
        self.assertStateAlmostEqual(qc, expected) 

        # Reset
        qc.px(0) # |10> (q1=1, q0=0)
        # q0 is 0, so no flip
        qc.cnot(0, 1)
        expected = np.zeros(4, dtype=complex)
        expected[2] = 1 # |10>
        self.assertStateAlmostEqual(qc, expected)

    def test_swap(self):
        qc = QuantumCircuit(3)
        # State |100> (q2=1, q1=0, q0=0) -> index 4
        qc.px(2)
        
        # SWAP(0, 2) -> |001> (q2=0, q1=0, q0=1) -> index 1
        qc.swap(0, 2)
        
        expected = np.zeros(8, dtype=complex)
        expected[1] = 1
        self.assertStateAlmostEqual(qc, expected)

    def test_complex_circuit(self):
        # H(0) -> CNOT(0, 1) -> Y(1)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cnot(0, 1)
        # State: 1/rt2 (|00> + |11>)
        
        qc.py(1)
        # Y on q1.
        # Y|0> = i|1>, Y|1> = -i|0>
        # |00> (q1=0) -> i|10>
        # |11> (q1=1) -> -i|01>
        # Result: 1/rt2 (i|10> -i|01>)
        # |10> is index 2 (q1=1, q0=0)
        # |01> is index 1 (q1=0, q0=1)
        
        expected = np.zeros(4, dtype=complex)
        expected[2] = 1j / np.sqrt(2)
        expected[1] = -1j / np.sqrt(2)
        
        self.assertStateAlmostEqual(qc, expected)

    def test_ccnot(self):
        qc = QuantumCircuit(3)
        qc.px(0)
        qc.px(1)
        # State |011> (q2=0, q1=1, q0=1)
        
        qc.ccnot(0, 1, 2)
        # Should flip q2 -> |111>
        expected = np.zeros(8, dtype=complex)
        expected[7] = 1
        self.assertStateAlmostEqual(qc, expected)

    def test_rotations(self):
        qc = QuantumCircuit(1)
        # Rx(pi) = -iX ~ |1> (global phase)
        qc.rx(0, np.pi)
        # Rx(pi) = [[0, -i], [-i, 0]]
        # |0> -> -i|1>
        expected = np.array([0, -1j])
        self.assertStateAlmostEqual(qc, expected)

    def test_phase(self):
        qc = QuantumCircuit(1)
        qc.px(0) # |1>
        qc.phase(0, np.pi/2) # i phase to |1>
        expected = np.array([0, 1j])
        self.assertStateAlmostEqual(qc, expected)

if __name__ == '__main__':
    unittest.main()
