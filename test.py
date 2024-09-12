import unittest
from quantipy import QuantumCircuit
import cmath

HALF_SQRT = complex((1 / 2) ** 0.5)
COMPLEX_ZERO = complex(0)
COMPLEX_ONE = complex(1)
INITIAL_STATE = [COMPLEX_ONE, COMPLEX_ZERO]
ZERO_STATE = INITIAL_STATE
ONE_STATE  = [COMPLEX_ZERO, COMPLEX_ONE]



class TestQuantipy(unittest.TestCase):
    """
    testing hadamard:
    applying it once results quibits being HALF_SQRT
    aPplying it twice results in quibits reverting to their original state
    """

    def test_hadamard(self):
        qc = QuantumCircuit(1)
        # storing states for later comparison
        zeroeth = qc.qubits[0][0]
        oneth = qc.qubits[0][1]
        qc.HADAMARD(0)
        self.assertAlmostEqual(HALF_SQRT, qc.qubits[0][0])
        self.assertAlmostEqual(HALF_SQRT, qc.qubits[0][1])
        qc.HADAMARD(0)
        self.assertAlmostEqual(zeroeth, qc.qubits[0][0])
        self.assertAlmostEqual(oneth, qc.qubits[0][1])

    def test_paulliX(self):
        qc = QuantumCircuit(1)

        qc.PaulliX(0)
        self.assertAlmostEqual(0, qc.qubits[0][0])
        self.assertAlmostEqual(1, qc.qubits[0][1])
        qc.PaulliX(0)
        self.assertAlmostEqual(1, qc.qubits[0][0])
        self.assertAlmostEqual(0, qc.qubits[0][1])

    def test_paulliY(self):
        qc = QuantumCircuit(2)
        qc.PaulliX(1)
        qc.PaulliY(0)
        self.assertAlmostEqual(0, qc.qubits[0][0])
        self.assertAlmostEqual(1j, qc.qubits[0][1])
        qc.PaulliY(1)
        self.assertAlmostEqual(-1j, qc.qubits[1][0])
        self.assertAlmostEqual(0, qc.qubits[1][1])

    def test_paulliZ(self):
        qc = QuantumCircuit(2)
        qc.PaulliX(1)
        qc.PaulliZ(0)
        self.assertAlmostEqual(1, qc.qubits[0][0])
        self.assertAlmostEqual(0, qc.qubits[0][1])
        qc.PaulliZ(1)
        self.assertAlmostEqual(0, qc.qubits[1][0])
        self.assertAlmostEqual(-1, qc.qubits[1][1])

    def test_PHASE_noPauliX(self):
        qc = QuantumCircuit(5)
        pi_values = [0.0, cmath.pi / 2, cmath.pi, (1.5) * cmath.pi, 2 * cmath.pi]
        for i in range(5):
            qc.PHASE(i, pi_values[i])
            self.assertAlmostEqual(qc.qubits[i][1], COMPLEX_ZERO)

    def test_PHASE_PauliX(self):
        qc = QuantumCircuit(5)
        pi_values = [0.0, cmath.pi / 2, cmath.pi, (1.5) * cmath.pi, 2 * cmath.pi]
        # i added some changes to PHASE
        # if theta % pi then use the real part only (-1 or 1)
        # else if theta % cmath.pi/2 then use the imaginary part only (-1j or 1j)

        euler_values = [
            COMPLEX_ONE,
            complex(0 + 1j),
            -COMPLEX_ONE,
            complex(0 - 1j),
            COMPLEX_ONE,
        ]
        for i in range(5):
            qc.PaulliX(i)
            qc.PHASE(i, pi_values[i])
            self.assertAlmostEqual(qc.qubits[i][1].real, euler_values[i].real)

        qc = QuantumCircuit(360)
        for i in range(360, 0, -1):
            qc.PaulliX(i - 1)
            qc.PHASE(i - 1, (2 * (cmath.pi / i)))
            self.assertAlmostEqual(
                qc.qubits[i - 1][1], cmath.exp(1j * (2 * (cmath.pi / i)))
            )

    def test_measure(self):
        qc = QuantumCircuit(100)
        for i in range(100):
            self.assertEqual(qc.measure(i), 0)

        qc = QuantumCircuit(100)
        for i in range(100):
            qc.PaulliX(i)
            self.assertEqual(qc.measure(i), 1)


    def test_CNOT(self):
        """flips target qubit only if control qubitis one"""
        qc = QuantumCircuit(2)
        qc.CNOT(0, 1)
        self.assertEqual(qc.qubits[1], INITIAL_STATE)
        qc.PaulliX(0)
        qc.CNOT(0, 1)
        self.assertEqual(qc.qubits[1], ONE_STATE)
        qc.CNOT(1, 0)
        self.assertEqual(qc.qubits[0], ZERO_STATE)

