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

    def create_bell_state(self):
        """
        Create a Bell state (maximally entangled state) for two qubits.
        |Φ+⟩ = (|00⟩ + |11⟩) / √2
        """
        if self.num_qubits != 2:
            raise ValueError("Bell state requires exactly 2 qubits")
        self.amplitudes[0] = 1 / np.sqrt(2)  # |00>
        self.amplitudes[3] = 1 / np.sqrt(2)  # |11>

    def measure_qubit(self, qubit_index):
        """
        Perform a measurement on a specific qubit and return the result (0 or 1).
        This collapses the state vector according to the measurement outcome.
        """
        if qubit_index < 0 or qubit_index >= self.num_qubits:
            raise ValueError("Invalid qubit index")

        # Calculate probabilities for the qubit being in state |0⟩ or |1⟩
        prob_0 = sum(self.get_probability(i) for i in range(self.num_states) if (i & (1 << qubit_index)) == 0)
        prob_1 = 1 - prob_0

        # Randomly choose an outcome based on the probabilities
        outcome = np.random.choice([0, 1], p=[prob_0, prob_1])

        # Collapse the state vector
        new_amplitudes = np.zeros(self.num_states, dtype=complex)
        normalization_factor = 0

        for i in range(self.num_states):
            if (i & (1 << qubit_index)) == (outcome << qubit_index):
                new_amplitudes[i] = self.amplitudes[i]
                normalization_factor += abs(self.amplitudes[i])**2

        self.amplitudes = new_amplitudes / np.sqrt(normalization_factor)

        return outcome

    def partial_trace(self, qubit_index):
        """
        Perform a partial trace over a specific qubit, returning the reduced density matrix.
        """
        if qubit_index < 0 or qubit_index >= self.num_qubits:
            raise ValueError("Invalid qubit index")

        reduced_dim = 2**(self.num_qubits - 1)
        rho = np.zeros((reduced_dim, reduced_dim), dtype=complex)

        for i in range(self.num_states):
            i_reduced = i & ~(1 << qubit_index)
            for j in range(self.num_states):
                j_reduced = j & ~(1 << qubit_index)
                if (i & (1 << qubit_index)) == (j & (1 << qubit_index)):
                    rho[i_reduced, j_reduced] += self.amplitudes[i] * np.conj(self.amplitudes[j])

        return rho

# Example usage
qsv = QuantumStateVector(2)
qsv.create_bell_state()
print("Bell state:")
print(qsv)

print("\nMeasuring first qubit:")
result = qsv.measure_qubit(0)
print(f"Measurement result: {result}")
print("State after measurement:")
print(qsv)

print("\nReduced density matrix for second qubit:")
rho = qsv.partial_trace(0)
print(rho)
