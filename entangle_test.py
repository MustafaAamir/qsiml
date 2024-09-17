import cmath
import random
import numpy as np
from typing import List, Tuple

class QuantumCircuit:
    def __init__(self, n: int = 1):
        self.num_qubits = n
        self.state_vector = np.zeros(2**n, dtype=complex)
        self.state_vector[0] = 1.0  # Initialize to |00...0>
        self.circuit = []

    def apply_gate(self, gate: np.ndarray, target_qubits: List[int]):
        # Create the full operator
        full_operator = np.eye(2**self.num_qubits, dtype=complex)

        # Calculate the matrix for the operation
        op_matrix = gate
        for i in range(self.num_qubits - 1, -1, -1):
            if i not in target_qubits:
                op_matrix = np.kron(op_matrix, np.eye(2))
            elif len(target_qubits) > 1:
                op_matrix = np.kron(op_matrix, gate)
                break

        # Apply the operator to the state vector
        self.state_vector = np.dot(op_matrix, self.state_vector)

    def h(self, i: int):
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self.apply_gate(H, [i])
        self.circuit.append(("H", [i]))

    def cnot(self, control: int, target: int):
        CNOT = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
        self.apply_gate(CNOT, [control, target])
        self.circuit.append(("CNOT", [control, target]))

    def measure(self, i: int) -> int:
        probabilities = np.abs(self.state_vector)**2
        outcome = np.random.choice(len(probabilities), p=probabilities)
        bit_string = format(outcome, f'0{self.num_qubits}b')
        measured_bit = int(bit_string[-(i+1)])

        # Collapse the state
        new_state = np.zeros_like(self.state_vector)
        for j, amplitude in enumerate(self.state_vector):
            if format(j, f'0{self.num_qubits}b')[-(i+1)] == str(measured_bit):
                new_state[j] = amplitude
        self.state_vector = new_state / np.linalg.norm(new_state)

        self.circuit.append(("M", [i]))
        return measured_bit

    def dump(self):
        probabilities = np.abs(self.state_vector)**2
        for i, prob in enumerate(probabilities):
            if prob > 1e-6:  # Only print non-negligible probabilities
                state = format(i, f'0{self.num_qubits}b')
                phase = cmath.phase(self.state_vector[i])
                print(f"|{state}> : {prob:.4f} (phase: {phase:.4f})")

    def draw(self):
        for i in range(self.num_qubits):
            line = f"q{i}: |0> "
            for gate, qubits in self.circuit:
                if i in qubits:
                    if gate == "H":
                        line += "-H-"
                    elif gate == "CNOT":
                        if i == qubits[0]:
                            line += "-●-"
                        else:
                            line += "-⊕-"
                    elif gate == "M":
                        line += "-M-"
                else:
                    line += "---"
            print(line)

# Example usage
qc = QuantumCircuit(2)
qc.h(0)
qc.cnot(0, 1)
qc.draw()
print("\nState vector:")
qc.dump()
