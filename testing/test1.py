import numpy as np

class QuantumStateVector:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.num_states = 2**num_qubits
        self.amplitudes = np.zeros(self.num_states, dtype=complex)

    def set_amplitude(self, basis_state, amplitude):
        if basis_state < 0 or basis_state >= self.num_states:
            raise ValueError("Invalid basis state")
        self.amplitudes[basis_state] = amplitude

    def get_amplitude(self, basis_state):
        if basis_state < 0 or basis_state >= self.num_states:
            raise ValueError("Invalid basis state")
        return self.amplitudes[basis_state]

    def get_probability(self, basis_state):
        amplitude = self.get_amplitude(basis_state)
        return np.abs(amplitude)**2

    def get_phase(self, basis_state):
        amplitude = self.get_amplitude(basis_state)
        return np.angle(amplitude)

    def normalize(self):
        norm = np.linalg.norm(self.amplitudes)
        if norm == 0:
            raise ValueError("Cannot normalize zero state vector")
        self.amplitudes /= norm

    def __str__(self):
        return "\n".join([f"|{i:0{self.num_qubits}b}>: {amp}" for i, amp in enumerate(self.amplitudes)])

# Example usage
qsv = QuantumStateVector(2)
qsv.set_amplitude(0, 1/np.sqrt(2))  # |00>
qsv.set_amplitude(3, 1j/np.sqrt(2))  # |11>
print(qsv)
print(f"Probability of |00>: {qsv.get_probability(0)}")
print(f"Probability of |01>: {qsv.get_probability(1)}")
print(f"Probability of |10>: {qsv.get_probability(2)}")
print(f"Probability of |11>: {qsv.get_probability(3)}")
