import random
import cmath
from typing import List,Tuple

HALF_SQRT = complex((1 / 2) ** 0.5)
COMPLEX_ZERO = complex(0)
COMPLEX_ONE = complex(1)
INITIAL_STATE = [COMPLEX_ONE, COMPLEX_ZERO]
ZERO_STATE = INITIAL_STATE
ONE_STATE = [COMPLEX_ZERO, COMPLEX_ONE]
ONE_SQRT2 = [complex(1 / cmath.sqrt(2)), complex(1 / cmath.sqrt(2))]


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
        self.circuit_operations: List[Tuple[str, List[int]]] = []

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
        self.circuit_operations.append(('X', [i]))

    def py(self, i):
        """
         [[0,-i]
        [i,0]]
        """
        self.qubits[i] = [self.qubits[i][1] * -1j, self.qubits[i][0] * 1j]
        self.circuit_operations.append(('Y', [i]))

    def pz(self, i):
        """
         [[1,0]
        [0,-1]]
        """
        self.qubits[i] = [self.qubits[i][0], self.qubits[i][1] * -1]
        self.circuit_operations.append(('Z', [i]))

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
        self.circuit_operations.append(('Rx', [i]))

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
        self.circuit_operations.append(('Ry', [i]))

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
        self.circuit_operations.append(('Rz', [i]))

    def phase(self, i, theta):
        self.qubits[i] = [
            self.qubits[i][0],
            complex(self.qubits[i][1] * (cmath.exp(1j * theta))),
        ]
        if theta % cmath.pi == 0:
            self.qubits[i][1] = self.qubits[i][1].real
        elif (theta % (cmath.pi / 2)) == 0:
            self.qubits[i][1] = (self.qubits[i][1].imag) * 1j
        self.circuit_operations.append(('P', [i]))

    def swap(self, i, j):
        temp = self.qubits[i]
        self.qubits[i] = self.qubits[j]
        self.qubits[j] = temp
        self.circuit_operations.append(('SWAP', [i]))

    def cnot(self, i, j):
        if self.qubits[i] == [0, 1]:
            self.px(j)
        self.circuit_operations.append(('CNOT', [i]))

    def h(self, i):
        self.qubits[i] = [
            (1 / (2**0.5)) * (self.qubits[i][1] + self.qubits[i][0]),
            (1 / (2**0.5)) * ((-1 * self.qubits[i][1]) + self.qubits[i][0]),
        ]
        self.circuit_operations.append(('H', [i]))

    def cswap(self, i, j, k):
        """
        Swaps target qubits j and k, given that qubit i has a one state
        Arguments:
            i (int) : index of the control bit
            j, k (int) : indexes of the target bits
        """
        if self.qubits[i] == [0, 1]:
            self.swap(j, k)
        self.circuit_operations.append(('CSWAP', [i]))

    def ccnot(self, i, j, k):
        """ """
        if self.qubits[i] == [0, 1] and self.qubits[j] == [0, 1]:
            self.px(k)
        self.circuit_operations.append(('CCNOT', [i]))

    def probability(self, i):
        pzero = self.qubits[i][0].real ** 2
        pone = self.qubits[i][1].real ** 2
        tp = pzero + pone
        pzero = pzero / tp
        pone = pone / tp
        return pzero, pone

    def measure(self, i):
        pzero, pone = self.probability(i)
        random_float = random.random()
        if random_float < pzero:
            ret = 0
            self.qubits[i] = [complex(1), complex(0)]
        else:
            ret = 1
            self.qubits[i] = [complex(0), complex(1)]
        self.circuit_operations.append(('M', [i]))
        return ret

    def measure_all(self):
        for i in range(len(self.qubits)):
            self.measure(i)

    def dump(self, msg: str = ""):
        """dumps info about the ith qubit without measuring its state"""
        """NEEDS REFACTORING"""
        """I'm banking on a numpy rewrite tho, so all good at the moment"""
        print(msg)
        state_vector = [1]
        for qubit in self.qubits:
            nstate_vector = []
            for amp in state_vector:
                 nstate_vector.append(amp * qubit[0])
                 nstate_vector.append(amp * qubit[1])
            state_vector = nstate_vector

        normalize = 0
        for amp in state_vector:
            normalize += (amp.real)**2
        normalize = cmath.sqrt(normalize)
        normalized_state_vector = []
        for amp in state_vector:
            normalized_state_vector.append(amp / normalize)

        for i, amp in enumerate(normalized_state_vector):
            prob = amp.real**2
            if prob < 1e-7:
                continue
            basis_state = bin(i)[2:]
            phase = cmath.phase(amp)
            print(f"basis state: |{basis_state}⟩\n  amplitude: {amp}\n  probability: {"{:.2f}".format(prob * 100)}\n  phase: {"{:.5f}".format(phase)}\n")



    def reset(self, i):
        """Qubit must be reset"""
        self.qubits[i] = INITIAL_STATE


qc = QuantumCircuit(2)
qc.px(0)
qc.h(0)
qc.cnot(0, 1)
qc.dump()
