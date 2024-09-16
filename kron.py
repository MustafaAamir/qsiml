import numpy as np

ONE_SQRT2 = complex(1/(2)**0.5)
"""
kronecker product works
wahali work on entanglement
"""
a = np.array([0, 1])
b = np.array([ONE_SQRT2, -ONE_SQRT2])
c = np.array([1, 0])
d = np.array([1, 0])
print(np.kron(np.kron(np.kron(a, b), c), d))

