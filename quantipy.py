import random
import cmath
from collections import Counter
import matplotlib.pyplot as plt
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
    def PaulliX(self, i):
        """
         [[0,1]
        [1,0]]
        """
        self.qubits[i] = [self.qubits[i][1], self.qubits[i][0]]

    def PaulliY(self, i):
        """
         [[0,-i]
        [i,0]]
        """
        self.qubits[i] = [self.qubits[i][1] * -1j, self.qubits[i][0] * 1j]

    def PaulliZ(self, i):
        """
         [[1,0]
        [0,-1]]
        """
        self.qubits[i] = [self.qubits[i][0], self.qubits[i][1] * -1]

    # Rotation Gates
    def Rx(self, i, theta):
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

    def Ry(self, i, theta):
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

    def Rz(self, i, theta):
        """
        [[exp(-i*theta/2),0]
        [0,[exp(i*theta/2)]]

        |0⟩ cool arrow
        """
        self.qubits[i] = [
            self.qubits[i][0] * cmath.exp(-1j * (theta / 2)),
            self.qubits[i][1] * cmath.exp(1j * (theta / 2)),
        ]

    def PHASE(self, i, theta):
        self.qubits[i] = [
            self.qubits[i][0],
            complex(self.qubits[i][1] * (cmath.exp(1j * theta))),
        ]
        if theta % cmath.pi == 0:
            self.qubits[i][1] = self.qubits[i][1].real
        elif (theta % (cmath.pi / 2)) == 0:
            self.qubits[i][1] = (self.qubits[i][1].imag) * 1j

    def SWAP(self, i, j):
        temp = self.qubits[i]
        self.qubits[i] = self.qubits[j]
        self.qubits[j] = temp

    def CNOT(self, i, j):
        if self.qubits[i] == [0, 1]:
            self.PaulliX(j)

    def HADAMARD(self, i):
        self.qubits[i] = [
            (1 / (2**0.5)) * (self.qubits[i][1] + self.qubits[i][0]),
            (1 / (2**0.5)) * ((-1 * self.qubits[i][1]) + self.qubits[i][0]),
        ]

    def CSWAP(self, i, j, k):
        """
        Swaps target qubits j and k, given that qubit i has a one state
        Arguments:
            i (int) : index of the control bit
            j, k (int) : indexes of the target bits
        """
        if self.qubits[i] == [0, 1]:
            self.SWAP(j, k)

    def CCNOT(self, i, j, k):
        """
        """
        if self.qubits[i] == [0, 1] and self.qubits[j] == [0, 1]:
            self.PaulliX(k)

    def measure(self, i):
        pzero = self.qubits[i][0].real ** 2
        pone = self.qubits[i][1].real ** 2
        # total probability should be one i think

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


def quantum_rng(bits):
    qc = QuantumCircuit(bits)
    number = 0
    for i in range(bits):
        qc.HADAMARD(i)
        result = qc.measure(i)
        number |= result << i
    return number


def random_plot():
    qc = QuantumCircuit(10)
    for i in range(10):
        qc.HADAMARD(i)
        print(qc.measure(i))

    # Generate a 8-bit random number
    random_number = quantum_rng(8)
    print(f"8-bit Quantum Random Number: {random_number} (binary: {random_number:08b})")

    # Generate multiple 8-bit random numbers
    num_samples = 1000
    samples = [quantum_rng(8) for _ in range(num_samples)]
    counts = Counter(samples)
    plt.bar(list(counts.keys()), list(counts.values()))
    plt.title("Distribution of Quantum Random Numbers")
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.show()

    mean = sum(samples) / num_samples
    variance = sum((x - mean) ** 2 for x in samples) / num_samples
    print(f"Mean: {mean:.2f}")
    print(f"Variance: {variance:.2f}")

qc = QuantumCircuit(2)
qc.HADAMARD(0)
ret = qc.measure(0)
qc.CNOT(0, 1)
print(qc)
