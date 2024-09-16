from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np

qc = QuantumCircuit(10)
qc.swap(2, 7)
qc.rx(0.029894449283742693, 5)
qc.ccx(8, 6, 0)
qc.cx(3, 5)
qc.cx(8, 3)
qc.ccx(8, 7, 6)
qc.cswap(0, 1, 6)
qc.swap(0, 6)
qc.ry(2.961675126617143, 8)
qc.cx(0, 1)
qc.rx(1.5976798923553746, 4)
qc.cswap(1, 5, 2)
qc.ry(0.8473061135504489, 0)
qc.cswap(3, 9, 0)
qc.ccx(2, 3, 6)
qc.swap(4, 0)
qc.swap(0, 9)
qc.ry(0.861107914305094, 0)
qc.rz(0.5157573130213783, 1)
qc.cswap(7, 8, 2)

print(qc.draw())
