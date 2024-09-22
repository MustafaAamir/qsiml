from qsiml import QuantumCircuit
import numpy as np

qc = QuantumCircuit(1)
qc.phase(0, np.pi/2)

qc.dump()
qc.draw()
