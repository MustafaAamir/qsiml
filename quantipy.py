import math
class Qubit:
    def __init__(self, zero,one):
        self.superposition=[zero,one]

def PaulliX(Qubit):
    Qubit.superposition=[Qubit.superposition[1],Qubit.superposition[0]]

def PaulliY(Qubit):
    Qubit.superposition=[Qubit.superposition[1]*-1j,Qubit.superposition[0]*1j]

def PaulliZ(Qubit):
    Qubit.superposition=[Qubit.superposition[0],Qubit.superposition[1]*-1]

def Phase(Qubit, theta):
    print (theta/math.pi,-1**0.5)
    Qubit.superposition=[Qubit.superposition[0],Qubit.superposition[1]*(-1**(theta/math.pi))]

def Swap(QubitA,QubitB):
    temp=QubitA.superposition
    QubitA.superposition=QubitB.superposition
    QubitB.superposition=temp

def CNOT(QubitA,QubitB):
    if QubitA.superposition==[0,1]:
        PaulliX(QubitB)

def Hamdard(Qubit):
    Qubit.superposition=[(1/(2**0.5))*(Qubit.superposition[1]+Qubit.superposition[0]),(1/(2**0.5))*((-1*Qubit.superposition[1])+Qubit.superposition[0])]


A=Qubit(0,1)
Phase(A,math.pi*0.5)
print(A.superposition)
