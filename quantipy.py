import random
import cmath
from typing import List


class QuantumCircuit:
    """
    A class containing qubits and gate operations that may be performed on qubits

    Attributes:
        qubits (List[List[complex]]): A list of qubits, which are 2x1 column vectors
    """

    def __init__(self, n=1):
        """
        Initializes qubits with n qubits with a zero state [1, 0]
        Parameters:
            n (int): The number of qubits to initialize (1 by default)
        """
        self.qubits: List[List[complex]] = [[complex(1), complex(0)] for _ in range(n)]

    def __repr__(self):
        str_qubits = ""
        for i, qubit in enumerate(self.qubits):
            str_qubits += f"Quibit {i}: [{qubit[0]}, {qubit[1]}]\n"
        return str_qubits

    # PaulliGates
    def px(self, i):
        """
         [[0,1]
        [1,0]]
        """
        self.qubits[i] = [self.qubits[i][1], self.qubits[i][0]]

    def py(self, i):
        """
         [[0,-i]
        [i,0]]
        """
        self.qubits[i] = [self.qubits[i][1] * -1j, self.qubits[i][0] * 1j]

    def pz(self, i):
        """
         [[1,0]
        [0,-1]]
        """
        self.qubits[i] = [self.qubits[i][0], self.qubits[i][1] * -1]

    # Rotation Gates
    def rx(self, i, theta):
        """
        [[cos(theta/2),   -i*sin(theta/2)],
        [-i*sin(theta/2),  cos(theta/2)]]
        """
        self.qubits[i] = [
            self.qubits[i][0] * cmath.cos(theta / 2)
            + self.qubits[i][1] * cmath.sin(theta / 2) * -1j,
            self.qubits[i][1] * cmath.cos(theta / 2)
            + self.qubits[i][0] * cmath.sin(theta / 2) * -1j,
        ]

    def ry(self, i, theta):
        """
        [[cos(theta/2),-sin(theta/2)]
        [sin(theta/2),cos(theta/2)]]
        """
        self.qubits[i] = [
            self.qubits[i][0] * cmath.cos(theta / 2)
            + self.qubits[i][1] * cmath.sin(theta / 2) * -1,
            self.qubits[i][1] * cmath.cos(theta / 2)
            + self.qubits[i][0] * cmath.sin(theta / 2),
        ]

    def rz(self, i, theta):
        """
        [[exp(-i*theta/2),0]
        [0,[exp(i*theta/2)]]

        |0⟩ cool arrow
        """
        self.qubits[i] = [
            self.qubits[i][0] * cmath.exp(-1j * (theta / 2)),
            self.qubits[i][1] * cmath.exp(1j * (theta / 2)),
        ]

    def phase(self, i, theta):
        self.qubits[i] = [
            self.qubits[i][0],
            complex(self.qubits[i][1] * (cmath.exp(1j * theta))),
        ]
        if theta % cmath.pi == 0:
            self.qubits[i][1] = self.qubits[i][1].real
        elif (theta % (cmath.pi / 2)) == 0:
            self.qubits[i][1] = (self.qubits[i][1].imag) * 1j

    def swap(self, i, j):
        temp = self.qubits[i]
        self.qubits[i] = self.qubits[j]
        self.qubits[j] = temp

    def cnot(self, i, j):
        if self.qubits[i] == [0, 1]:
            self.px(j)

    def h(self, i):
        self.qubits[i] = [
            (1 / (2**0.5)) * (self.qubits[i][1] + self.qubits[i][0]),
            (1 / (2**0.5)) * ((-1 * self.qubits[i][1]) + self.qubits[i][0]),
        ]

    def cswap(self, i, j, k):
        """
        Swaps target qubits j and k, given that qubit i has a one state
        Arguments:
            i (int) : index of the control bit
            j, k (int) : indexes of the target bits
        """
        if self.qubits[i] == [0, 1]:
            self.swap(j, k)

    def ccnot(self, i, j, k):
        """ """
        if self.qubits[i] == [0, 1] and self.qubits[j] == [0, 1]:
            self.px(k)

    def measure(self, i):
        pzero = self.qubits[i][0].real ** 2
        pone = self.qubits[i][1].real ** 2
        tp = pzero + pone
        pzero = pzero / tp
        pone = pone / tp
        random_float = random.random()
        if random_float < pzero:
            ret = 0
            self.qubits[i] = [complex(1), complex(0)]
        else:
            ret = 1
            self.qubits[i] = [complex(0), complex(1)]
        return ret
