from qiskit import QuantumCircuit
from qiskit.circuit.quantumcircuit import StandardGate
from qiskit.quantum_info import Statevector
from qiskit.visualization.state_visualization import state_drawer
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

qc = QuantumCircuit(10)
qc.swap(2, 7)
qc.rx(0.029894, 5)
qc.ccx(8, 6, 0)
qc.cx(3, 5)
qc.cx(8, 3)
qc.ccx(8, 7, 6)
qc.cswap(0, 1, 6)
qc.swap(0, 6)
qc.ry(2.961675, 8)
qc.cx(0, 1)
qc.rx(1.597679, 4)
qc.cswap(1, 5, 2)
qc.ry(0.847306, 0)
qc.cswap(3, 9, 0)
qc.ccx(2, 3, 6)
qc.swap(4, 0)
qc.swap(0, 9)
qc.ry(0.861107, 0)
qc.rz(0.515757, 1)
qc.cswap(7, 8, 2)

print(qc.data)
