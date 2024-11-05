![image](https://github.com/user-attachments/assets/80a3a50c-1268-4b34-9053-edb0f06ccb12)

Qsiml is a Python-based quantum computing simulator that provides a minimalist approach to quantum circuit simulation.

## Installation

```bash
pip install qsiml
```

### Quantum Circuit

A quantum circuit is represented by the `QuantumCircuit` class. It manages a collection of qubits and applies quantum gates to manipulate their states.

```python
from qsiml import QuantumCircuit

qc = QuantumCircuit(n)  # Creates a circuit with `n` qubits
```

### Gates

### Single-Qubit Gates

1. Hadamard (H): Creates superposition
   ```python
   qc.h(qubit)
   ```

2. Pauli-X (NOT): Bit flip
   ```python
   qc.px(qubit)
   ```

3. Pauli-Y: Rotation around Y-axis
   ```python
   qc.py(qubit)
   ```

4. Pauli-Z: Phase flip
   ```python
   qc.pz(qubit)
   ```

5. Phase (P): Applies a phase shift
   ```python
   qc.phase(qubit, theta)
   ```

6. Rotation Gates: Rotate around X, Y, or Z axis
   ```python
   qc.rx(qubit, theta)
   qc.ry(qubit, theta)
   qc.rz(qubit, theta)
   ```
   where θ is the rotation angle in radians.

### Multi-Qubit Gates

1. CNOT: Controlled-NOT
   ```python
   qc.cnot(control, target)
   ```

2. SWAP: Swaps two qubits
   ```python
   qc.swap(qubit1, qubit2)
   ```

3. Toffoli (CCNOT): Controlled-Controlled-NOT
   ```python
   qc.ccnot(control1, control2, target)
   ```

4. Fredkin (CSWAP): Controlled-SWAP
   ```python
   qc.cswap(control, target1, target2)
   ```

## Measurement

Measure all qubits, collapsing the state vector:

```python
result = qc.measure_all() # collapses the state vector to a single basis states
# returns a bitstring of the basis state and stores the collapsed state in qc.classical_bits
```

Measure a specific qubit, partially collapsing the state vector.

```python
qc.measure(qubit) # classical state of qn is stored in qc.classical_bits[n]
```

## Circuit Visualization

```python
from qsiml import QuantumCircuit

qc = QuantumCircuit(5)
qc.px(0)
qc.h(1)
qc.h(2)
qc.h(3)
qc.ccnot(1, 2, 3)
qc.ccnot(2, 3, 4)
```

Display the circuit as an ASCII diagram:

```python
qc.draw("Circuit Visualization: ")
```
```
Circuit Visualization

|q0⟩—X————————————————

|q1⟩————H————————●————
                 │
|q2⟩———————H—————●——●—
                 │  │
|q3⟩——————————H——⨁——●—
                    │
|q4⟩————————————————⨁—
```

```python
qc.operations("Operations: ")
```
prints the gates applied with respect to time:
```
  Operations:
    1. X on qubit 0
    2. H on qubit 1
    3. H on qubit 2
    4. H on qubit 3
    5. CCNOT on qubits 1, 2, 3
    6. CCNOT on qubits 2, 3, 4
```

```python
print(qc.circuit)
```
```
prints the internal circuit representation

[('X', [0]), ('H', [1]), ('H', [2]), ('H', [3]), ('CCNOT', [1, 2, 3]), ('CCNOT', [2, 3, 4])]
```

## State Inspection

View the circuit's state without collapsing it.

```python
qc.dump("Dump table: ")
```

prints a table which shows the amplitude, probability, and phase of each possible basis state.
```
Dump Table:
+---------------+---------------+----------------------+---------+
| Basis State   | Probability   | Amplitude            |   Phase |
+===============+===============+======================+=========+
| |00001⟩       | 12.500000%    | 0.353553 + 0.000000i |       0 |
| |00011⟩       | 12.500000%    | 0.353553 + 0.000000i |       0 |
| |00101⟩       | 12.500000%    | 0.353553 + 0.000000i |       0 |
| |00111⟩       | 12.500000%    | 0.353553 + 0.000000i |       0 |
| |01001⟩       | 12.500000%    | 0.353553 + 0.000000i |       0 |
| |01011⟩       | 12.500000%    | 0.353553 + 0.000000i |       0 |
| |11101⟩       | 12.500000%    | 0.353553 + 0.000000i |       0 |
| |11111⟩       | 12.500000%    | 0.353553 + 0.000000i |       0 |
+---------------+---------------+----------------------+---------+
```

## Examples

### Bell State Preparation

```python
qc = QuantumCircuit(2)
qc.h(0)
qc.cnot(0, 1)
qc.draw("Bell State diagram: ")
qc.dump("Bell State dump table: ")
```

Output:
```
Bell State diagram:
|q0⟩—H——●—
        │
|q1⟩————⨁—

Bell State dump table:
+---------------+---------------+----------------------+---------+
| Basis State   | Probability   | Amplitude            |   Phase |
+===============+===============+======================+=========+
| |00⟩          | 50.000000%    | 0.707107 + 0.000000i |       0 |
| |11⟩          | 50.000000%    | 0.707107 + 0.000000i |       0 |
+---------------+---------------+----------------------+---------+
```

### Quantum Fourier Transform (2 qubits)

```python
qc = QuantumCircuit(2)
qc.h(0)
qc.phase(1, np.pi/2)
qc.cnot(0, 1)
qc.h(1)
qc.swap(0, 1)
qc.draw("Draw: ")
qc.dump("Dump: ")
```

```
Draw:

|q0⟩—H—————————————●—————x—
                   │     │
|q1⟩————-P(1.5707)—⨁——H——x—

Dump:

+---------------+---------------+-----------------------+---------+
| Basis State   | Probability   | Amplitude             |   Phase |
+===============+===============+=======================+=========+
| |00⟩          | 56.250000%    | 0.750000 + 0.000000i  | 0       |
| |01⟩          | 56.250000%    | 0.750000 + 0.000000i  | 0       |
| |10⟩          | 31.250000%    | 0.250000 + 0.500000i  | 1.10715 |
| |11⟩          | 31.250000%    | -0.250000 + 0.500000i | 2.03444 |
+---------------+---------------+-----------------------+---------+
```

### Theory for nerds

Quantum computing leverages the principles of quantum mechanics to perform computations. Unlike classical bits, which can be in one of two states (0 or 1), quantum bits (qubits) can exist in a superposition of states, represented as a linear combination of basis states:

`|ψ⟩ = α|0⟩ + β|1⟩`

where `α` and `β` are complex numbers satisfying `|α⟩^2 + |β⟩^2 = 1.0`

A trivial example to illustrate the, albeit niche, advantage of quantum computing over classical computing is the Deutsch-Jozsa algorithm. In the problem, we're given a black box quantum computer known as an oracle that implements some function `f: {0, 1}ⁿ-> {0, 1}`, which takes an n-bit binary value as input and returns either a 0 or a 1 for each input. The function output is either constant, either 1 OR 0 for all inputs, or balanced, 0 for exactly half of the input domain and 1 for the other half. The task is to determine if `f` is constant or balanced using the function.

the deterministic classical approach requires `2^(n - 1) + 1` evaluations to prove that f is either constant or balanced. It needs to map *half + 1* the set of inputs to evaluate, with 100% certainty, the nature of the oracle. If `n := 2`:

|**x (input)** | **f(x) (output)** |
|:--------:|:------------:|
| 00       |    0         |
| 01       |    0         |
| 10       |    1         |
| 00       |    1         |

Only the first 3 calculations are required to determine that the oracle is balanced. Though, the computational complexity increases exponentially, which makes it more expensive to solve for larger values of `n`.
This is where quantum computing shines. The Deutsch-Jozsa algorithm applies the oracle to a superposition of all possible inputs, represented by `n + 1`, where the first `n` qubits are initialized to |0⟩, and the last one is initialized to |1⟩.

```python
n = 10
qc = QuantumCircuit(n + 1) # initialize a circuit with n + 1 qubits
qc.px(n) # initialize the last qubit to |1⟩
```
Apply the Hadamard gate to all qubits to create a superposition of all possible states (try it!)
```python
# applies the hadamard gate to all qubits in the system
for i in range(n + 1):
    qc.h(i)
```
The next step is to create an oracle. The oracle essentially acts as a query system, which is easy to represent in classical computing by storing the mapped value in a certain memory register. In quantum computing however, this is impractical. We'll have to create a custom quantum circuit representation of an oracle. We'll use the `n + 1`th qubit as an ancilla qubit that is initialized to a state of |1⟩, and the first `n` qubits as the query. For a balanced function, the oracle should flip the ancilla qubit for exactly half of the input states.

```python
import numpy as np
random_bits = np.random.randint(1, 2**n) # returns a random integer between 1 and 2**n - 1 inclusive.
for i in range(n):
    # applies cnot with control bits that lie within the randomly generated binary number. If `random_bit` = `101`, then qubits 0 and 2 would be used as control bits.
    if a & (1 << i):
        qc.cnot(i, n)
```

Afterwards, we revert the query qubits back to their original state by applying the hadamard gate

```python
for i in range(n):
    qc.h(i)
```

Finally, we measure the query qubits individually
```python
for i in range(n):
    qc.measure(i)
```
The measured values of the nth qubit are stored in `qc.classical_bits[n]`. If all measured values are 0, i.e. `qc.classical_bits[0..n]`, then the oracle is a constant function. Anything other than that, the oracle is a balanced function.

Now that a balanced oracle function has been implemented, we can implement a constant oracle.

```python
from qsiml import QuantumCircuit
import numpy as np

class DeutschJozsa():
    def __init__(self, n: int = 10):
        self.qc = QuantumCircuit(n + 1)
        self.n = n

    def constant_oracle(self, constant_value: int):
        if constant_value == 0:
            self.qc.i(self.n)
        else:
            self.qc.px(self.n)

    def balanced_oracle(self, random_bits: int):
        for i in range(self.n):
            if random_bits & (1 << i):
                self.qc.cnot(i, self.n)

    def deutsch_jozsa(self):
        n = self.n
        constant_or_balanced = np.random.randint(0, 2)
        constant_value = np.random.randint(0, 2)
        random_bits = np.random.randint(1, 2**n)

        self.qc.px(n)
        for i in range(n + 1):
            self.qc.h(i)

        if constant_or_balanced == 0:
            self.constant_oracle(constant_value)
        else:
            self.balanced_oracle(random_bits)

        for i in range(n):
            self.qc.h(i)

        for i in range(n):
            self.qc.measure(i)


        self.qc.draw()
        print("Classical Bits: ", self.qc.classical_bits[:-1])

dj = DeutschJozsa(10)
dj.deutsch_jozsa()
```

returns this for a constant oracle (Notice how every measured value is 0):
```
|q00⟩—H——————————————————————————————————————H—————————————————————————————M————————————————————————————
                                                                           0
|q01⟩————H——————————————————————————————————————H—————————————————————————————M—————————————————————————
                                                                              0
|q02⟩———————H——————————————————————————————————————H—————————————————————————————M——————————————————————
                                                                                 0
|q03⟩——————————H——————————————————————————————————————H—————————————————————————————M———————————————————
                                                                                    0
|q04⟩—————————————H——————————————————————————————————————H—————————————————————————————M————————————————
                                                                                       0
|q05⟩————————————————H——————————————————————————————————————H—————————————————————————————M—————————————
                                                                                          0
|q06⟩———————————————————H——————————————————————————————————————H—————————————————————————————M——————————
                                                                                             0
|q07⟩——————————————————————H——————————————————————————————————————H—————————————————————————————M———————
                                                                                                0
|q08⟩—————————————————————————H——————————————————————————————————————H—————————————————————————————M————
                                                                                                   0
|q09⟩————————————————————————————H——————————————————————————————————————H—————————————————————————————M—
                                                                                                      0
|q10⟩———————————————————————————————X——H——X—————————————————————————————————————————————————————————————

Classical Bits: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

And this for a balanced oracle (The measured values form a non-zero bitstring)
```
|q00⟩—H——————————————————————————————————————————————————H—————————————————————————————M————————————————————————————
                                                                                       0
|q01⟩————H————————————————————————————————●—————————————————H—————————————————————————————M—————————————————————————
                                          │                                               1
|q02⟩———————H—————————————————————————————│————————————————————H—————————————————————————————M——————————————————————
                                          │                                                  0
|q03⟩——————————H——————————————————————————│———————————————————————H—————————————————————————————M———————————————————
                                          │                                                     0
|q04⟩—————————————H———————————————————————│——————————————————————————H—————————————————————————————M————————————————
                                          │                                                        0
|q05⟩————————————————H————————————————————│—————————————————————————————H—————————————————————————————M—————————————
                                          │                                                           0
|q06⟩———————————————————H—————————————————│——●—————————————————————————————H—————————————————————————————M——————————
                                          │  │                                                           1
|q07⟩——————————————————————H——————————————│——│——●—————————————————————————————H—————————————————————————————M———————
                                          │  │  │                                                           1
|q08⟩—————————————————————————H———————————│——│——│——●—————————————————————————————H—————————————————————————————M————
                                          │  │  │  │                                                           1
|q09⟩————————————————————————————H————————│——│——│——│——●—————————————————————————————H—————————————————————————————M—
                                          │  │  │  │  │                                                           1
|q10⟩———————————————————————————————X——H——⨁——⨁——⨁——⨁——⨁—————————————————————————————————————————————————————————————

Classical Bits: [0, 1, 0, 0, 0, 0, 1, 1, 1, 1]
```
You can import this class using:

```python
from qsiml import DeutschJozsa
```

### State Vector Representation

In Qsiml, an n-qubit system is represented by a 2^n dimensional complex vector, known as the state vector. For example, a two-qubit system is represented by a 4-dimensional vector:

`|ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩`

where `|α|^2 + |β|^2 + |γ|^2 + |δ|^2 = 1`.

