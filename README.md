# Qsiml

A fast, minimal implementation of a quantum computing simulator in python


# Installation

```bash
pip install qsiml
```

# Usage

Import the QuantumCircuit class
```python
from qsiml import QuantumCircuit
```
Instantiate a QuantumCircuit class
```python
qc = QuantumCircuit(3)
```
Apply gates
```python
qc.h(0)
qc.px(1)
qc.rx(2,np.pi)
qc.ccnot(0, 1, 2)
```
Dump all possible states
```python
qc.dump()
```
Collapse the circuit to one of the possible basis states
```python
qc.measure_all()
```
Draw the circuit representation
```python
qc.draw()
```

