<a id="quantipy"></a>

# quantipy

Wins:
![image](https://github.com/user-attachments/assets/00eb0e2f-7bd8-4a9e-a251-0897b8d79407)

<a id="quantipy.QuantumCircuit"></a>

## QuantumCircuit Objects

```python
class QuantumCircuit()
```

A class representing a quantum circuit with qubits and gate operations.

This class allows for the creation and manipulation of quantum circuits,
including various quantum gates and measurement operations.

**Attributes**:

- `qubits` _List[List[complex]]_ - A list of qubits, where each qubit is represented as a 2x1 column vector of complex numbers.
- `circuit` _List[Tuple[str, List[int]]]_ - A list of operations performed on the circuit, where each operation is a tuple of the gate name and the qubit indices it acts upon.
- `qubits_count` _int_ - Number of qubits in the circuit

<a id="quantipy.QuantumCircuit.__init__"></a>

#### \_\_init\_\_

```python
def __init__(n: int = 1)
```

Initializes qubits with n qubits with a zero state [1, 0]

**Arguments**:

- `n` _int, optional_ - The number of qubits to initialize (1 by default)

<a id="quantipy.QuantumCircuit.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

Return a string representation of every qubit's amplitude in the following format:
    Qubit {i}: [alpha, beta]
where alpha and beta are the individual probability amplitudes for each state.

<a id="quantipy.QuantumCircuit.px"></a>

#### px

```python
def px(i: int)
```

Apply the Pauli-X gate (NOT gate) to the i-th qubit.
The Pauli-X gate flips the state of the qubit, transforming |0⟩ to |1⟩ and vice versa.

**Arguments**:

- `i` _int_ - The index of the qubit to apply the gate to.


**Raises**:

  IndexError if i < 0 or i > self.qubits_count

<a id="quantipy.QuantumCircuit.py"></a>

#### py

```python
def py(i: int)
```

Apply the Pauli-Y gate to the i-th qubit.
The Pauli-Y gate rotates the qubit state around the Y-axis of the Bloch sphere by π radians.

**Arguments**:

- `i` _int_ - The index of the qubit to apply the gate to.


**Raises**:

  IndexError if i < 0 or i > self.qubits_count

<a id="quantipy.QuantumCircuit.pz"></a>

#### pz

```python
def pz(i: int)
```

Apply the Pauli-Z gate to the i-th qubit.
The Pauli-Z gate rotates the qubit state around the Z-axis of the Bloch sphere by π radians.

**Arguments**:

- `i` _int_ - The index of the qubit to apply the gate to.


**Raises**:

  IndexError if i < 0 or i > self.qubits_count

<a id="quantipy.QuantumCircuit.rx"></a>

#### rx

```python
def rx(i: int, theta: float)
```

Apply the rx(θ) to the i-th qubit.
The rx gate rotates the qubit state around the X-axis of the Bloch sphere by theta radians.

**Arguments**:

- `i` _int_ - The index of the qubit to apply the gate to.
- `theta` _float_ - The angle of rotation in radians.


**Raises**:

  IndexError if i < 0 or i > self.qubits_count

<a id="quantipy.QuantumCircuit.ry"></a>

#### ry

```python
def ry(i: int, theta: float)
```

Apply the rx(θ) to the i-th qubit.
The ry gate rotates the qubit state around the Y-axis of the Bloch sphere by theta radians.

**Arguments**:

- `i` _int_ - The index of the qubit to apply the gate to.
- `theta` _float_ - The angle of rotation in radians.


**Raises**:

  IndexError if i < 0 or i > self.qubits_count

<a id="quantipy.QuantumCircuit.rz"></a>

#### rz

```python
def rz(i: int, theta: float)
```

Apply the rz(θ) to the i-th qubit.
The rz gate rotates the qubit state around the Z-axis of the Bloch sphere by theta radians.

**Arguments**:

- `i` _int_ - The index of the qubit to apply the gate to.
- `theta` _float_ - The angle of rotation in radians.


**Raises**:

  IndexError if i < 0 or i > self.qubits_count

<a id="quantipy.QuantumCircuit.phase"></a>

#### phase

```python
def phase(i: int, theta: float)
```

Apply a phase shift to the i-th qubit.
phase(θ) adds a phase e^(i*θ) to the |1⟩, leaving |0⟩ unchanged.

**Arguments**:

- `i` _int_ - The index of the qubit to apply the gate to.
- `theta` _float_ - The phase angle in radians.


**Raises**:

  IndexError if i < 0 or i > self.qubits_count

<a id="quantipy.QuantumCircuit.swap"></a>

#### swap

```python
def swap(i: int, j: int)
```

Swap the states of two qubits.

**Arguments**:

- `i` _int_ - The index of the first qubit.
- `j` _int_ - The index of the second qubit.


**Raises**:

  IndexError if i < 0 or i > self.qubits_count
  IndexError if j < 0 or j > self.qubits_count

  ValueError if i and j aren't distinct

<a id="quantipy.QuantumCircuit.cnot"></a>

#### cnot

```python
def cnot(i: int, j: int)
```

Apply the CNOT gate with qubit i as control and qubit j as target.
If the control qubit is |1⟩, the amplitudes of the target qubit are flipped.

**Arguments**:

- `i` _int_ - The index of the control qubit.
- `j` _int_ - The index of the target qubit.


**Raises**:

  IndexError if i < 0 or i > self.qubits_count
  IndexError if j < 0 or j > self.qubits_count

  ValueError if i and j aren't distinct

<a id="quantipy.QuantumCircuit.h"></a>

#### h

```python
def h(i: int)
```

Apply the Hadamard (H) gate to the i-th qubit.
The H gate creates an equal superposition of |0⟩ and |1⟩ states.

**Arguments**:

- `i` _int_ - The index of the qubit to apply the gate to.


**Raises**:

  IndexError if i < 0 or i > self.qubits_count

<a id="quantipy.QuantumCircuit.cswap"></a>

#### cswap

```python
def cswap(i: int, j: int, k: int)
```

Apply a CSWAP (Fredkin) gate.
Swaps the amplitudes of qubits j and k if qubit i is in the |1⟩ state.

**Arguments**:

- `i` _int_ - The index of the control qubit.
- `j` _int_ - The index of the first target qubit.
- `k` _int_ - The index of the second target qubit.


**Raises**:

  IndexError if i < 0 or i > self.qubits_count
  IndexError if j < 0 or j > self.qubits_count
  IndexError if k < 0 or k> self.qubits_count


  ValueError if i, j, and k aren't distinct

<a id="quantipy.QuantumCircuit.ccnot"></a>

#### ccnot

```python
def ccnot(i: int, j: int, k: int)
```

Apply a CCNOT (Toffoli) gate.
Swaps the amplitudes of qubit k if both qubits i and j are in the |1⟩ state.

**Arguments**:

- `i` _int_ - The index of the first control qubit.
- `j` _int_ - The index of the second control qubit.
- `k` _int_ - The index of the target qubit.



**Raises**:

  IndexError if i < 0 or i > self.qubits_count
  IndexError if j < 0 or j > self.qubits_count
  IndexError if k < 0 or k > self.qubits_count

  ValueError if i, j and k aren't distinct

<a id="quantipy.QuantumCircuit.probability"></a>

#### probability

```python
def probability(i: int) -> Tuple[float, float]
```

Calculate the probability of measuring the i-th qubit in the |0⟩ and |1⟩ states.

**Arguments**:

- `i` _int_ - The index of the qubit.


**Returns**:

  Tuple[float, float]: A tuple containing the probabilities (p_zero, p_one).


**Raises**:

  IndexError if i < 0 or i > self.qubits_count

<a id="quantipy.QuantumCircuit.measure"></a>

#### measure

```python
def measure(i: int) -> int
```

Performs a measurement on the i-th qubit.
This collapses the qubit's state to either |0⟩ or |1⟩ based on its current probabilities.

**Arguments**:

- `i` _int_ - The index of the qubit to measure.


**Returns**:

- `int` - The result of the measurement (0 or 1).


**Raises**:

  IndexError if i < 0 or i > self.qubits_count

<a id="quantipy.QuantumCircuit.measure_all"></a>

#### measure\_all

```python
def measure_all() -> List[int]
```

Performs a measurement on every qubit in the circuit.
This collapses the state of all qubit to either |0⟩ or |1⟩ based on their current probabilities.

**Returns**:

- `List[int]` - The results of the measurement (0 or 1) of all qubits in the circuit.

<a id="quantipy.QuantumCircuit.dump"></a>

#### dump

```python
def dump(msg: str = "")
```

Print the current state of the quantum circuit without affecting it.
Displays the probability amplitudes for each basis state along with their probabilities and phases.

**Arguments**:

- `msg` _str, optional_ - An optional message to print before the state dump. Defaults to "".

<a id="quantipy.QuantumCircuit.reset"></a>

#### reset

```python
def reset(i)
```

Reset the i-th qubit to the |0⟩ state.

**Arguments**:

- `i` _int_ - The index of the qubit to reset.


**Raises**:

  IndexError if i < 0 or i > self.qubits_count

<a id="quantipy.QuantumCircuit.draw"></a>

#### draw

```python
def draw(header: str = "")
```

Print an ASCII representation of the quantum circuit.

**Arguments**:

- `header` _str, optional_ - An optional header to print above the circuit representation. Defaults to "".

<a id="quantipy.QuantumCircuit.operations"></a>

#### operations

```python
def operations(header: str = "")
```

Prints the gates applied to each qubit(s) in order.

**Arguments**:

- `msg` _str, optional_ - An optional header to print above the description. Defaults to "".

