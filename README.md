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

### Gate Operations

Qsiml supports a variety of quantum gates, each with its own unique operation. Here's a detailed explanation of how each gate works:

### Hadamard Gate (H)

The Hadamard gate, denoted by h(qubit), applies a Hadamard transformation to the specified qubit. The Hadamard transformation is a linear transformation that takes a qubit from the state |0to a superposition of |0and |1, denoted by:

|ψ= 1/√2 (|0+ |1)

In other words, the Hadamard gate creates a superposition of the qubit's state, with equal probability of measuring 0 or 1.

### Pauli-X Gate (X)

The Pauli-X gate, denoted by px(qubit), applies a bit flip to the specified qubit. The Pauli-X gate is equivalent to a NOT gate in classical computing, and it flips the qubit's state from |0to |1or vice versa.

### Pauli-Y Gate (Y)

The Pauli-Y gate, denoted by py(qubit), applies a rotation around the Y-axis of the Bloch sphere to the specified qubit. The Pauli-Y gate is equivalent to a rotation of π/2 radians around the Y-axis, and it changes the qubit's state from |0to i|1or vice versa.

### Pauli-Z Gate (Z)

The Pauli-Z gate, denoted by pz(qubit), applies a rotation around the Z-axis of the Bloch sphere to the specified qubit. The Pauli-Z gate is equivalent to a rotation of π/2 radians around the Z-axis, and it changes the qubit's state from |0to -i|1or vice versa.

### Rotation Gates (RX, RY, RZ)

The rotation gates, denoted by rx(qubit, angle), ry(qubit, angle), and rz(qubit, angle), apply a rotation around the X, Y, or Z-axis of the Bloch sphere to the specified qubit, respectively. The rotation angle is specified in radians.

### Phase Gate (P)

The phase gate, denoted by phase(qubit, angle), applies a phase shift to the specified qubit. The phase shift is equivalent to a rotation around the Z-axis of the Bloch sphere, and it changes the qubit's state from |0to e^(i*angle)|0or vice versa.

### Swap Gate (SWAP)

The swap gate, denoted by swap(qubit1, qubit2), swaps the states of two qubits.

### Controlled-NOT Gate (CNOT)

The controlled-NOT gate, denoted by cnot(control, target), applies a NOT gate to the target qubit if the control qubit is in the state |1.


### Controlled-CNOT Gate (CCNOT)

The controlled-NOT gate, denoted by ccnot(control1, control2, target), applies a NOT gate to the target qubit if both the control qubits are in the state |1.

### Controlled-SWAP Gate (CSWAP)

The controlled-NOT gate, denoted by cswap(control, target1, target2), applies a SWAP gate to the target qubits if the control qubit is in the state |1.


### Dump Function

The dump() function is used to print all possible states of the quantum circuit in a human-readable format. The dump function works by iterating over all possible states of the qubits and printing the corresponding state vector.

### Example Code:
```python
qc = QuantumCircuit(2)
qc.h(0)
qc.cnot(0, 1)
qc.dump()
```
Output:
```
+---------------+---------------+----------------------+---------+
| Basis State   | Probability   | Amplitude            |   Phase |
+===============+===============+======================+=========+
| |000⟩         | 50.000000%    | 0.707107 + 0.000000i |       0 |
| |011⟩         | 50.000000%    | 0.707107 + 0.000000i |       0 |
+---------------+---------------+----------------------+---------+
```
This shows that the circuit is in a superposition of all four possible states, with equal probability of measuring each state.

### Draw Function

The draw() function is used to visualize the quantum circuit as a text-based diagram. The draw function works by iterating over the gates in the circuit and printing a corresponding symbol for each gate.

### Example Code:

```python
qc = QuantumCircuit(2)
qc.h(0)
qc.cnot(0, 1)
qc.draw()

```
Output:
```

|q0⟩—H——●—
        │
|q1⟩————⨁—

|q2⟩——————

```
This shows the Hadamard gate applied to qubit 0, followed by the controlled-NOT gate applied to qubits 0 and 1.
