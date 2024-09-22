from qsiml import QuantumCircuit
import numpy as np


def dj():
    oracleType, oracleValue = np.random.randint(2), np.random.randint(2)

    n = 10
    if oracleType == 0:
        print("The oracle returns a constant value", oracleValue)
    else:
        print("The oracle return a balanced function")
        a = np.random.randint(1, 2**n)

    qc = QuantumCircuit(n + 1)
    for i in range(n):
        qc.h(i)

    qc.px(n)
    qc.h(n)

    if oracleType == 0:
        if oracleValue == 1:
            qc.px(n)
        else:
            qc.i(n)
    else:
        for i in range(n):
            if a & (1 << i):
                qc.cnot(i, n)

    for i in range(n):
        qc.h(i)


    for i in range(n):
        qc.measure(i)

    qc.dump()
    qc.draw()

if __name__ == "__main__":
    dj()
