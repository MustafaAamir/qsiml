import random
import cmath
from collections import Counter
import matplotlib.pyplot as plt


class QuantumCircuit:
    def __init__(self,n=1):
        self.qubits=[[1,0] for _ in range(n)]

    def __repr__(self):
        str_qubits = ""
        for i, qubit in enumerate(self.qubits):
            str_qubits += f"Quibit {i}: [{qubit[0]}, {qubit[1]}]\n"
        return str_qubits

    def PaulliX(self,i):
        self.qubits[i]=[self.qubits[i][1],self.qubits[i][0]]

    def PaulliY(self,i):
        self.qubits[i]=[self.qubits[i][1]*-1j,self.qubits[i][0]*1j]

    def PaulliZ(self,i):
        self.qubits[i]=[self.qubits[i][0],self.qubits[i][1]*-1]


    def PHASE(self,i, theta):
        self.qubits[i].superposition=[
                self.qubits[i][0],
                complex(self.qubits[i][1]*(cmath.exp(1j * theta)))
                ]
        if (theta % cmath.pi == 0):
            self.qubits[i][1] = self.qubits[i][1].real

    def SWAP(self, i, j):
        temp=self.qubits[i]
        self.qubits[i]=self.qubits[j]
        self.qubits[j]=temp

    def CNOT(self, i, j):
        if self.qubits[i]==[0,1]:
            self.PaulliX(self.qubits[j])

    def HADAMARD(self,i):
        self.qubits[i]=[
                (1/(2**0.5))*(self.qubits[i][1]+self.qubits[i][0]),
                (1/(2**0.5))*((-1*self.qubits[i][1])+self.qubits[i][0])
                ]

    def CSWAP(self,i,j,k):
        if self.qubits[i]==[0,1]:
            self.SWAP(j,k)

    def CCNOT(self,i,j,k):
        if self.qubits[i]==[0,1] and self.qubits[j]==[0,1]:
            self.PaulliX(k)
    def measure(self, i):
        pzero = self.qubits[i][0].real ** 2
        pone  = self.qubits[i][1].real ** 2
        # total probability should be one i think

        tp = pzero + pone
        pzero = pzero / tp
        pone  = pone / tp

        random_float = random.random()
        if random_float < pzero:
            ret = 0
            self.qubits[i] = [1, 0]
        else:
            ret = 1
            self.qubits[i]= [0, 1]
        return ret



def quantum_rng(bits):
    qc = QuantumCircuit(bits)
    number = 0
    for i in range(bits):
        qc.HADAMARD(i)
        result = qc.measure(i)
        number |= (result << i)
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
