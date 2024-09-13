import random
import cmath
import math
from typing import List, Union, Dict, Tuple

class QuantumCircuit:
    def __init__(self, n=1):
        self.num_qubits = n
        self.qubits: List[List[complex]] = [[complex(1), complex(0)] for _ in range(n)]
        self.circuit_operations: List[Tuple[str, int, Union[int, float]]] = []

    def __repr__(self):
        return "\n".join([f"Qubit {i}: [{qubit[0]}, {qubit[1]}]" for i, qubit in enumerate(self.qubits)])

    def apply_gate(self, gate_matrix: List[List[complex]], target: int):
        """Apply a single-qubit gate to the target qubit."""
        self.qubits[target] = [
            gate_matrix[0][0] * self.qubits[target][0] + gate_matrix[0][1] * self.qubits[target][1],
            gate_matrix[1][0] * self.qubits[target][0] + gate_matrix[1][1] * self.qubits[target][1]
        ]

    # Single-qubit gates
    def PaulliX(self, target: int):
        self.apply_gate([[0, 1], [1, 0]], target)
        self.circuit_operations.append(('X', target, None))

    def PaulliY(self, target: int):
        self.apply_gate([[0, -1j], [1j, 0]], target)
        self.circuit_operations.append(('Y', target, None))

    def PaulliZ(self, target: int):
        self.apply_gate([[1, 0], [0, -1]], target)
        self.circuit_operations.append(('Z', target, None))

    def HADAMARD(self, target: int):
        self.apply_gate([[1/math.sqrt(2), 1/math.sqrt(2)], [1/math.sqrt(2), -1/math.sqrt(2)]], target)
        self.circuit_operations.append(('H', target, None))

    def PHASE(self, target: int, theta: float):
        self.apply_gate([[1, 0], [0, cmath.exp(1j * theta)]], target)
        self.circuit_operations.append(('P', target, theta))

    def Rx(self, target: int, theta: float):
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        self.apply_gate([[cos, -1j * sin], [-1j * sin, cos]], target)
        self.circuit_operations.append(('Rx', target, theta))

    def Ry(self, target: int, theta: float):
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        self.apply_gate([[cos, -sin], [sin, cos]], target)
        self.circuit_operations.append(('Ry', target, theta))

    def Rz(self, target: int, theta: float):
        self.apply_gate([[cmath.exp(-1j * theta / 2), 0], [0, cmath.exp(1j * theta / 2)]], target)
        self.circuit_operations.append(('Rz', target, theta))

    # Multi-qubit gates
    def CNOT(self, control: int, target: int):
        if self.qubits[control][1] != 0:
            self.PaulliX(target)
        self.circuit_operations.append(('CNOT', control, target))

    def SWAP(self, a: int, b: int):
        self.qubits[a], self.qubits[b] = self.qubits[b], self.qubits[a]
        self.circuit_operations.append(('SWAP', a, b))

    def CSWAP(self, control: int, a: int, b: int):
        if self.qubits[control][1] != 0:
            self.SWAP(a, b)
        self.circuit_operations.append(('CSWAP', control, (a, b)))

    def CCNOT(self, control1: int, control2: int, target: int):
        if self.qubits[control1][1] != 0 and self.qubits[control2][1] != 0:
            self.PaulliX(target)
        self.circuit_operations.append(('CCNOT', (control1, control2), target))

    def measure(self, target: int) -> int:
        prob_zero = abs(self.qubits[target][0]) ** 2
        result = 0 if random.random() < prob_zero else 1
        self.qubits[target] = [complex(1-result), complex(result)]
        self.circuit_operations.append(('M', target, None))
        return result

    # ... (other methods remain unchanged) ...

    def print_circuit(self):
        """Print an ASCII representation of the quantum circuit."""
        circuit_diagram = [['-'] * (len(self.circuit_operations) * 3 + 1) for _ in range(self.num_qubits)]

        for i, op in enumerate(self.circuit_operations):
            gate, qubit, param = op
            col = i * 3 + 1

            if gate in ['X', 'Y', 'Z', 'H']:
                circuit_diagram[qubit][col] = gate
            elif gate in ['Rx', 'Ry', 'Rz']:
                circuit_diagram[qubit][col-1:col+2] = [gate[0], gate[1], '']
            elif gate == 'P':
                circuit_diagram[qubit][col] = 'P'
            elif gate == 'CNOT':
                control, target = qubit, param
                circuit_diagram[control][col] = '•'
                circuit_diagram[target][col] = '⊕'
                for q in range(min(control, target) + 1, max(control, target)):
                    circuit_diagram[q][col] = '|'
            elif gate == 'SWAP':
                a, b = qubit, param
                circuit_diagram[a][col] = '×'
                circuit_diagram[b][col] = '×'
                for q in range(min(a, b) + 1, max(a, b)):
                    circuit_diagram[q][col] = '|'
            elif gate == 'CSWAP':
                control, (a, b) = qubit, param
                circuit_diagram[control][col] = '•'
                circuit_diagram[a][col] = '×'
                circuit_diagram[b][col] = '×'
                for q in range(min(control, a, b) + 1, max(control, a, b)):
                    circuit_diagram[q][col] = '|'
            elif gate == 'CCNOT':
                control1, control2 = qubit
                target = param
                circuit_diagram[control1][col] = '•'
                circuit_diagram[control2][col] = '•'
                circuit_diagram[target][col] = '⊕'
                for q in range(min(control1, control2, target) + 1, max(control1, control2, target)):
                    circuit_diagram[q][col] = '|'
            elif gate == 'M':
                circuit_diagram[qubit][col-1:col+2] = ['M', 'E', 'A']

        # Print the circuit diagram
        for i, qubit_line in enumerate(circuit_diagram):
            print(f"q{i}: ", end='')
            print(''.join(qubit_line))

# Example usage
if __name__ == "__main__":
    qc = QuantumCircuit(3)
    qc.HADAMARD(0)
    qc.CNOT(0, 1)
    qc.Rx(2, math.pi/4)
    qc.CCNOT(0, 1, 2)
    qc.measure(0)
    qc.SWAP(1, 2)
    qc.CSWAP(0, 1, 2)

    print("Quantum Circuit ASCII Representation:")
    qc.print_circuit()

    print("\nFinal quantum state:")
    print(qc)
