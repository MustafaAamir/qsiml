import math
class Qubit:
    def __init__(self, zero=1,one=0):
        self.superposition=[zero,one]

class quantumCircuit:
    def __init__(self,n=1):
        self.qubits=[Qubit for i in range(n)]

    def PaulliX(self,i):
        self.qubits[i].superposition=[self.qubits[i].superposition[1],self.qubits[i].superposition[0]]

    def PaulliY(self,i):
        self.qubits[i].superposition=[self.qubits[i].superposition[1]*-1j,self.qubits[i].superposition[0]*1j]

    def PaulliZ(self,i):
        self.qubits[i].superposition=[self.qubits[i].superposition[0],self.qubits[i].superposition[1]*-1]

    def PHASE(self,i, theta):
        self.qubits[i].superposition=[self.qubits[i].superposition[0],self.qubits[i].superposition[1]*(-1**(theta/math.pi))]

    def SWAP(self,i,j,):
        temp=self.qubits[i].superposition
        self.qubits[i].superposition=self.qubits[j].superposition
        self.qubits[j].superposition=temp

    def CNOT(self,i,j):
        if self.qubits[i].superposition==[0,1]:
            self.PaulliX(self.qubits[j])

    def HADAMARD(self,i):
        self.qubits[i].superposition=[(1/(2**0.5))*(self.qubits[i].superposition[1]+self.qubits[i].superposition[0]),(1/(2**0.5))*((-1*self.qubits[i].superposition[1])+self.qubits[i].superposition[0])]

    def CSWAP(self,i,j,k):
        if self.qubits[i].superposition==[0,1]:
            self.SWAP(j,k)

    def CCNOT(self,i,j,k):
        if self.qubits[i].superposition==[0,1] and self.qubits[j].superposition==[0,1]:
            self.PaulliX(k)


