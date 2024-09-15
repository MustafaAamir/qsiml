import random
import cmath
from typing import List, Tuple

HALF_SQRT = complex((1 / 2) ** 0.5)
COMPLEX_ZERO = complex(0)
COMPLEX_ONE = complex(1)
INITIAL_STATE = [COMPLEX_ONE, COMPLEX_ZERO]
ZERO_STATE = INITIAL_STATE
ONE_STATE = [COMPLEX_ZERO, COMPLEX_ONE]
ONE_SQRT2 = [complex(1 / cmath.sqrt(2)), complex(1 / cmath.sqrt(2))]


def _check_index(i, qubits_count: int):
    if i < 0 or i > qubits_count - 1:
        raise IndexError(
            f"Qubit index '{i}' is out of range. Valid range is 0 to {qubits_count - 1}"
        )


def _check_distinct(args: List[int]):
    if len(args) != len(set(args)):
        raise ValueError("Arguments to need to be distinct")


class QuantumCircuit:
    """
    A class representing a quantum circuit with qubits and gate operations.

    This class allows for the creation and manipulation of quantum circuits,
    including various quantum gates and measurement operations.

    Attributes:
        qubits (List[List[complex]]): A list of qubits, where each qubit is represented as a 2x1 column vector of complex numbers.
        circuit (List[Tuple[str, List[int]]]): A list of operations performed on the circuit, where each operation is a tuple of the gate name and the qubit indices it acts upon.
        qubits_count (int): Number of qubits in the circuit
    """

    def __init__(self, n: int = 1):
        """
        Initializes qubits with n qubits with a zero state [1, 0]
        Args:
            n (int, optional): The number of qubits to initialize (1 by default)
        """
        self.qubits: List[List[complex]] = [INITIAL_STATE for _ in range(n)]
        self.qubits_count: int = n
        self.circuit: List[Tuple[str, List[int]]] = []

    def __repr__(self):
        """
        Return a string representation of every qubit's amplitude in the following format:
            Qubit {i}: [alpha, beta]
        where alpha and beta are the individual probability amplitudes for each state.
        """
        str_qubits = ""
        for i, qubit in enumerate(self.qubits):
            str_qubits += f"Quibit {i}: [{qubit[0]}, {qubit[1]}]\n"
        return str_qubits

        # PaulliGates

    def px(self, i: int):
        """
        Apply the Pauli-X gate (NOT gate) to the i-th qubit.
        The Pauli-X gate flips the state of the qubit, transforming |0⟩ to |1⟩ and vice versa.

        Args:
            i (int): The index of the qubit to apply the gate to.

        Raises:
            IndexError if i < 0 or i > self.qubits_count
        """
        _check_index(i, self.qubits_count)
        self.qubits[i] = [self.qubits[i][1], self.qubits[i][0]]
        self.circuit.append(("X", [i]))

    def py(self, i: int):
        """
        Apply the Pauli-Y gate to the i-th qubit.
        The Pauli-Y gate rotates the qubit state around the Y-axis of the Bloch sphere by π radians.

        Args:
            i (int): The index of the qubit to apply the gate to.

        Raises:
            IndexError if i < 0 or i > self.qubits_count

        """
        _check_index(i, self.qubits_count)
        self.qubits[i] = [self.qubits[i][1] * -1j, self.qubits[i][0] * 1j]
        self.circuit.append(("Y", [i]))

    def pz(self, i: int):
        """
        Apply the Pauli-Z gate to the i-th qubit.
        The Pauli-Z gate rotates the qubit state around the Z-axis of the Bloch sphere by π radians.

        Args:
            i (int): The index of the qubit to apply the gate to.

        Raises:
            IndexError if i < 0 or i > self.qubits_count
        """

        _check_index(i, self.qubits_count)
        self.qubits[i] = [self.qubits[i][0], self.qubits[i][1] * -1]
        self.circuit.append(("Z", [i]))

    # Rotation Gates
    def rx(self, i: int, theta: float):
        """
        Apply the rx(θ) to the i-th qubit.
        The rx gate rotates the qubit state around the X-axis of the Bloch sphere by theta radians.

        Args:
            i (int): The index of the qubit to apply the gate to.
            theta (float): The angle of rotation in radians.

        Raises:
            IndexError if i < 0 or i > self.qubits_count
        """

        _check_index(i, self.qubits_count)
        self.qubits[i] = [
            self.qubits[i][0] * cmath.cos(theta / 2)
            + self.qubits[i][1] * cmath.sin(theta / 2) * -1j,
            self.qubits[i][1] * cmath.cos(theta / 2)
            + self.qubits[i][0] * cmath.sin(theta / 2) * -1j,
        ]
        self.circuit.append(("Rx", [i]))

    def ry(self, i: int, theta: float):
        """
        Apply the rx(θ) to the i-th qubit.
        The ry gate rotates the qubit state around the Y-axis of the Bloch sphere by theta radians.
        Args:
            i (int): The index of the qubit to apply the gate to.
            theta (float): The angle of rotation in radians.

        Raises:
            IndexError if i < 0 or i > self.qubits_count
        """

        _check_index(i, self.qubits_count)
        self.qubits[i] = [
            self.qubits[i][0] * cmath.cos(theta / 2)
            + self.qubits[i][1] * cmath.sin(theta / 2) * -1,
            self.qubits[i][1] * cmath.cos(theta / 2)
            + self.qubits[i][0] * cmath.sin(theta / 2),
        ]
        self.circuit.append(("Ry", [i]))

    def rz(self, i: int, theta: float):
        """
        Apply the rz(θ) to the i-th qubit.
        The rz gate rotates the qubit state around the Z-axis of the Bloch sphere by theta radians.
        Args:
            i (int): The index of the qubit to apply the gate to.
            theta (float): The angle of rotation in radians.

        Raises:
            IndexError if i < 0 or i > self.qubits_count
        """
        _check_index(i, self.qubits_count)
        self.qubits[i] = [
            self.qubits[i][0] * cmath.exp(-1j * (theta / 2)),
            self.qubits[i][1] * cmath.exp(1j * (theta / 2)),
        ]
        self.circuit.append(("Rz", [i]))

    def phase(self, i: int, theta: float):
        """
        Apply a phase shift to the i-th qubit.
        phase(θ) adds a phase e^(i*θ) to the |1⟩, leaving |0⟩ unchanged.

        Args:
            i (int): The index of the qubit to apply the gate to.
            theta (float): The phase angle in radians.

        Raises:
            IndexError if i < 0 or i > self.qubits_count
        """
        _check_index(i, self.qubits_count)
        self.qubits[i] = [
            self.qubits[i][0],
            complex(self.qubits[i][1] * (cmath.exp(1j * theta))),
        ]
        if theta % cmath.pi == 0:
            self.qubits[i][1] = self.qubits[i][1].real
        elif (theta % (cmath.pi / 2)) == 0:
            self.qubits[i][1] = (self.qubits[i][1].imag) * 1j
        self.circuit.append(("P", [i]))

    def swap(self, i: int, j: int):
        """
        Swap the states of two qubits.

        Args:
            i (int): The index of the first qubit.
            j (int): The index of the second qubit.

        Raises:
            IndexError if i < 0 or i > self.qubits_count
            IndexError if j < 0 or j > self.qubits_count

            ValueError if i and j aren't distinct
        """
        _check_index(i, self.qubits_count)
        _check_index(j, self.qubits_count)
        _check_distinct([i, j])
        temp = self.qubits[i]
        self.qubits[i] = self.qubits[j]
        self.qubits[j] = temp
        self.circuit.append(("SWAP", [i, j]))

    def cnot(self, i: int, j: int):
        """
        Apply the CNOT gate with qubit i as control and qubit j as target.
        If the control qubit is |1⟩, the amplitudes of the target qubit are flipped.

        Args:
            i (int): The index of the control qubit.
            j (int): The index of the target qubit.

        Raises:
            IndexError if i < 0 or i > self.qubits_count
            IndexError if j < 0 or j > self.qubits_count

            ValueError if i and j aren't distinct
        """
        _check_distinct([i, j])
        _check_index(i, self.qubits_count)
        _check_index(j, self.qubits_count)
        if self.qubits[i] == ONE_STATE:
            self.qubits[j] = [self.qubits[j][1], self.qubits[j][0]]
        self.circuit.append(("CNOT", [i, j]))

    def h(self, i: int):
        """
        Apply the Hadamard (H) gate to the i-th qubit.
        The H gate creates an equal superposition of |0⟩ and |1⟩ states.
        Args:
            i (int): The index of the qubit to apply the gate to.

        Raises:
            IndexError if i < 0 or i > self.qubits_count
        """
        _check_index(i, self.qubits_count)
        self.qubits[i] = [
            (1 / (2**0.5)) * (self.qubits[i][1] + self.qubits[i][0]),
            (1 / (2**0.5)) * ((-1 * self.qubits[i][1]) + self.qubits[i][0]),
        ]
        self.circuit.append(("H", [i]))

    def cswap(self, i: int, j: int, k: int):
        """
        Apply a CSWAP (Fredkin) gate.
        Swaps the amplitudes of qubits j and k if qubit i is in the |1⟩ state.

        Args:
            i (int): The index of the control qubit.
            j (int): The index of the first target qubit.
            k (int): The index of the second target qubit.

        Raises:
            IndexError if i < 0 or i > self.qubits_count
            IndexError if j < 0 or j > self.qubits_count
            IndexError if k < 0 or k> self.qubits_count


            ValueError if i, j, and k aren't distinct
        """
        _check_distinct([i, j, k])
        _check_index(i, self.qubits_count)
        _check_index(j, self.qubits_count)
        _check_index(k, self.qubits_count)
        if self.qubits[i] == ONE_STATE:
            temp = self.qubits[j]
            self.qubits[j] = self.qubits[k]
            self.qubits[k] = temp

        self.circuit.append(("CSWAP", [i, j, k]))

    def ccnot(self, i: int, j: int, k: int):
        """
        Apply a CCNOT (Toffoli) gate.
        Swaps the amplitudes of qubit k if both qubits i and j are in the |1⟩ state.

        Args:
            i (int): The index of the first control qubit.
            j (int): The index of the second control qubit.
            k (int): The index of the target qubit.


        Raises:
            IndexError if i < 0 or i > self.qubits_count
            IndexError if j < 0 or j > self.qubits_count
            IndexError if k < 0 or k > self.qubits_count

            ValueError if i, j and k aren't distinct
        """
        _check_index(i, self.qubits_count)
        _check_index(j, self.qubits_count)
        _check_index(k, self.qubits_count)
        _check_distinct([i, j, k])
        if self.qubits[i] == ONE_STATE and self.qubits[j] == ONE_STATE:
            self.qubits[k] = [self.qubits[k][1], self.qubits[k][0]]
        self.circuit.append(("CCNOT", [i, j, k]))

    def probability(self, i: int) -> Tuple[float, float]:
        """
        Calculate the probability of measuring the i-th qubit in the |0⟩ and |1⟩ states.

        Args:
            i (int): The index of the qubit.

        Returns:
            Tuple[float, float]: A tuple containing the probabilities (p_zero, p_one).

        Raises:
            IndexError if i < 0 or i > self.qubits_count
        """

        _check_index(i, self.qubits_count)
        pzero = self.qubits[i][0].real ** 2
        pone = self.qubits[i][1].real ** 2
        tp = pzero + pone
        pzero = pzero / tp
        pone = pone / tp
        return pzero, pone

    def measure(self, i: int) -> int:
        """
        Performs a measurement on the i-th qubit.
        This collapses the qubit's state to either |0⟩ or |1⟩ based on its current probabilities.

        Args:
            i (int): The index of the qubit to measure.

        Returns:
            int: The result of the measurement (0 or 1).

        Raises:
            IndexError if i < 0 or i > self.qubits_count
        """
        _check_index(i, self.qubits_count)
        pzero, _ = self.probability(i)
        random_float = random.random()
        if random_float < pzero:
            ret = 0
            self.qubits[i] = ZERO_STATE
        else:
            ret = 1
            self.qubits[i] = ONE_STATE
        self.circuit.append(("M", [i]))
        return ret

    def measure_all(self) -> List[int]:
        """
        Performs a measurement on every qubit in the circuit.
        This collapses the state of all qubit to either |0⟩ or |1⟩ based on their current probabilities.

        Returns:
            List[int]: The results of the measurement (0 or 1) of all qubits in the circuit.
        """

        measured_values = []
        measured_values.append(self.measure(i) for i in range(self.qubits_count))
        return measured_values

    def dump(self, msg: str = ""):
        """
        Print the current state of the quantum circuit without affecting it.
        Displays the probability amplitudes for each basis state along with their probabilities and phases.

        Args:
            msg (str, optional): An optional message to print before the state dump. Defaults to "".
        """
        print(msg)
        state_vector = [1]
        for qubit in self.qubits:
            nstate_vector = []
            for amp in state_vector:
                nstate_vector.append(amp * qubit[0])
                nstate_vector.append(amp * qubit[1])
            state_vector = nstate_vector

        normalize = 0
        for amp in state_vector:
            normalize += (amp.real) ** 2
        normalize = cmath.sqrt(normalize)
        normalized_state_vector = []
        for amp in state_vector:
            normalized_state_vector.append(amp / normalize)

        for i, amp in enumerate(normalized_state_vector):
            prob = amp.real**2
            if prob < 1e-8:
                continue
            basis_state = bin(i)[2:]
            phase = cmath.phase(amp)
            print(
                f"basis state: |{basis_state}⟩\n  amplitude: {amp}\n  probability: {"{:.2f}".format(prob * 100)}\n  phase: {"{:.5f}".format(phase)}\n"
            )

    def reset(self, i):
        """
        Reset the i-th qubit to the |0⟩ state.

        Args:
            i (int): The index of the qubit to reset.

        Raises:
            IndexError if i < 0 or i > self.qubits_count
        """

        _check_index(i, self.qubits_count)
        self.qubits[i] = INITIAL_STATE


    def draw(self, header: str = ""):
        """
        Print an ASCII representation of the quantum circuit.

        Args:
            header (str, optional): An optional header to print above the circuit representation. Defaults to "".
        """
        circuit = self.circuit
        gate_symbols = {
            "H": "H",
            "X": "X",
            "Y": "Y",
            "M": "M",
            "Z": "Z",
            "CNOT": "⨁",
            "SWAP": "x",
            "CSWAP": "x",
            "Rx": "Rx",
            "Ry": "Rʏ",
            "Rz": "Rᴢ",
            "CCNOT": "⨁",
            # Add more gate symbols as needed
        }

        num_qubits = self.qubits_count
        num_gates = len(circuit)
        if header != "":
            print(header)
        for qubit in range(num_qubits):
            entangle=[' ' for i in  range(3*num_gates+4)]
            print(f"|q{qubit}⟩", end="")
            for gate_index in range(num_gates):
                gate, targets = circuit[gate_index]
                if qubit in targets:
                    if len(targets) > 1:
                        if qubit == targets[0]:
                            print("—●—", end="")
                        elif qubit == targets[-1]:
                            if gate == "SWAP":
                                print("—x—", end="")
                            else:
                                print(f"—{gate_symbols[gate]}—", end="")
                        else:
                            print("—●—", end="")
                        
                        if targets[-1]>targets[0]:
                            if qubit<targets[-1] and qubit>=targets[0]:
                                entangle[3*gate_index+5]="│"
                        else:
                            if qubit>=targets[-1] and qubit<targets[0]:
                                entangle[3*gate_index+5]="│"
                        if len(targets)==3:
                            if targets[-1]>targets[1]:
                                if qubit<targets[-1] and qubit>=targets[1]:
                                    entangle[3*gate_index+5]="│"
                            else:
                                if qubit>=targets[-1] and qubit<targets[1]:
                                    entangle[3*gate_index+5]="│"

                    else:
                        if gate in ("Rx", "Ry", "Rz"):
                            print(f"—{gate_symbols[gate]}", end="")
                            entangle[3*gate_index+5]="π"
                        else:
                            print(f"—{gate_symbols[gate]}—", end="")
                else:
                    if qubit<targets[-1] and qubit>=targets[0] or qubit>=targets[-1] and qubit<targets[0]:
                        print("—│—", end="")
                        entangle[3*gate_index+5]="│"
                    else:
                        print("———", end="")
            print()
            for i in range(len(entangle)):
          
                print(entangle[i],end='')
            
            print()

    def operations(self, header: str = ""):
        """
        Prints the gates applied to each qubit(s) in order.

        Args:
            msg (str, optional): An optional header to print above the description. Defaults to "".
        """
        print(header)
        for i, (gate, targets) in enumerate(self.circuit):
            qubit_plural = "qubit"
            if len(targets) > 1:
                qubit_plural += "s"
            target_str = ", ".join(map(str, targets))
            print(f"{i + 1}. {gate} on {qubit_plural} {target_str}")
