import math
import random


class quantumCircuit:
    def __init__(self,n=1):
        self.qubits=[[1,0] for i in range(n)]

    def PaulliX(self,i):
        self.qubits[i]=[self.qubits[i][1],self.qubits[i][0]]

    def PaulliY(self,i):
        self.qubits[i]=[self.qubits[i][1]*-1j,self.qubits[i][0]*1j]

    def PaulliZ(self,i):
        self.qubits[i]=[self.qubits[i][0],self.qubits[i][1]*-1]

    def PHASE(self,i, theta):
        self.qubits[i]=[self.qubits[i][0],self.qubits[i][1]*((-1)**(theta/math.pi))]

    def SWAP(self,i,j,):
        temp=self.qubits[i]
        self.qubits[i]=self.qubits[j]
        self.qubits[j]=temp

    def CNOT(self,i,j):
        if self.qubits[i]==[0,1]:
            self.PaulliX(self.qubits[j])

    def HADAMARD(self,i):
        self.qubits[i]=[(1/(2**0.5))*(self.qubits[i][1]+self.qubits[i][0]),(1/(2**0.5))*((-1*self.qubits[i][1])+self.qubits[i][0])]

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
    
