from quasi import QuantumCircuit, Qubit
import unittest
from typing import List, Tuple
import cmath
from random import randint, random


HALF_SQRT = complex((1 / 2) ** 0.5)
COMPLEX_ZERO = complex(0)
COMPLEX_ONE = complex(1)
INITIAL_STATE = [COMPLEX_ONE, COMPLEX_ZERO]
ZERO_STATE = INITIAL_STATE
ONE_STATE = [COMPLEX_ZERO, COMPLEX_ONE]
ONE_SQRT2 = [complex(1 / cmath.sqrt(2)), complex(1 / cmath.sqrt(2))]


class TestMeasureAll(unittest.TestCase):
    def test_hadamard(self):
        qc = QuantumCircuit(1)
        for _ in range(10):
            qc.h(0)
            self.assertIn(qc.measure_all(), [[0], [1]])
            qc.qubits[0].states = INITIAL_STATE

    def test_hadamard_cnot(self):
        qc = QuantumCircuit(2)
        for _ in range(10):
            qc.h(0)
            qc.cnot(0, 1)
            self.assertIn(qc.measure_all(), [[0, 0], [1, 1]])
            qc.qubits[0].states = INITIAL_STATE
            qc.qubits[1].states = INITIAL_STATE

    def test_hadamard_cnot_chain(self):
        qc = QuantumCircuit(10)
        qc.h(0)
        qc.cnot(0, 1)
        qc.cnot(1, 2)
        qc.cnot(2, 3)
        qc.cnot(3, 4)
        qc.cnot(4, 5)
        qc.cnot(5, 6)
        qc.cnot(6, 7)
        qc.cnot(7, 8)
        qc.cnot(8, 9)
        self.assertIn(qc.measure_all(), [[0] * 10, [1] * 10])

    def test_hadamards_cnots(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.cnot(0, 1)
        self.assertIn(qc.measure_all(), [[0, 0], [0, 1], [1, 0], [1, 1]])

    def test_cswap(self):
        for _ in range(10):
            qc = QuantumCircuit(3)
            qc.h(0)
            qc.px(1)
            qc.cswap(0, 1, 2)
            qc.cswap(0, 1, 2)
            measured = qc.measure_all()
            print(measured)

            self.assertIn(measured, [[0, 1, 0], [1, 1, 0]])

        for _ in range(10):
            qc = QuantumCircuit(3)

            qc.h(0)
            qc.px(1)
            qc.cswap(0, 1, 2)
            qc.cswap(0, 1, 2)
            qc.cswap(0, 1, 2)
            measured = qc.measure_all()
            print(measured)

            self.assertIn(measured, [[0, 1, 0], [1, 0, 1]])







