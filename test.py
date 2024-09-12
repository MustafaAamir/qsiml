import unittest
from quantipy import QuantumCircuit

HALF_SQRT = complex((1/2) ** 0.5)

class TestQuantipy(unittest.TestCase):
    '''
    testing hadamard:
    applying it once results quibits being HALF_SQRT
    aPplying it twice results in quibits reverting to their original state
    '''
    def test_hadamard(self):
        qc = QuantumCircuit(1)
        # storing states for later comparison
        zeroeth = qc.qubits[0][0]
        oneth   = qc.qubits[0][1]
        qc.HADAMARD(0)
        self.assertAlmostEqual(HALF_SQRT, qc.qubits[0][0])
        self.assertAlmostEqual(HALF_SQRT, qc.qubits[0][1])
        qc.HADAMARD(0)
        self.assertAlmostEqual(zeroeth, qc.qubits[0][0])
        self.assertAlmostEqual(oneth, qc.qubits[0][1])

