import numpy as np
from typing import List, Tuple
from tabulate import tabulate

HALF_SQRT = complex((1 / 2) ** 0.5)
COMPLEX_ZERO = complex(0)
COMPLEX_ONE = complex(1)
INITIAL_STATE = [COMPLEX_ONE, COMPLEX_ZERO]
ZERO_STATE = INITIAL_STATE
ONE_STATE = [COMPLEX_ZERO, COMPLEX_ONE]
ONE_SQRT2 = [complex(1 / np.sqrt(2)), complex(1 / np.sqrt(2))]

"""1 uqbit gates"""
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

PX = np.array([[0, 1], [1, 0]])

PY = np.array([[0, -1j], [1j, 0]])

PZ = np.array([[1, 0], [0, -1]])

NOT = np.array([[0, 1], [1, 0]])


def _check_index(i, qubits_count: int):
    if i < 0 or i > qubits_count - 1:
        raise IndexError(
            f"Qubit index '{i}' is out of range. Valid range is 0 to {qubits_count - 1}"
        )


def _check_distinct(args: List[int]):
    if len(args) != len(set(args)):
        raise ValueError("Arguments to need to be distinct")


class Qubit:
    def __init__(self):
        self.states: List[complex] = INITIAL_STATE


class QuantumCircuit:
    def __init__(self, n: int = 1):
        self.qubits: List[Qubit] = [Qubit() for _ in range(n)]
        self.qubits_count: int = n
        self.thetas: List[float] = []
        self.circuit: List[Tuple[str, List[int | float]]] = []
        self.state_vector = np.zeros(2**self.qubits_count, dtype=complex)
        self.evaluated = False

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
        self.circuit.append(("Rx", [i, theta]))
        self.thetas.append(theta)

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
        self.circuit.append(("Ry", [i, theta]))
        self.thetas.append(theta)

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
        self.circuit.append(("Rz", [i, theta]))
        self.thetas.append(theta)

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
        self.circuit.append(("P", [i, theta]))
        self.thetas.append(theta)

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
        self.circuit.append(("CCNOT", [i, j, k]))

    def reset(self, i):
        """
        Reset the i-th qubit to the |0⟩ state.

        Args:
            i (int): The index of the qubit to reset.

        Raises:
            IndexError if i < 0 or i > self.qubits_count
        """

        _check_index(i, self.qubits_count)
        self.qubits[i].states = INITIAL_STATE

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

    def _katas(self, header=""):
        print(header)
        length = self.qubits_count
        output_str = ""
        output_str += f"\t\tuse qs = Qubit[{length}];\n"
        for gate, targets in self.circuit:
            args = ""
            for arg in targets:
                args += f"qs[{arg}], "
            args = args[:-2]
            output_str += f"\t\t{gate}({args});\n"
        print(output_str)

    def apply_sqg(self, gate, qubit):
        n = 2**self.qubits_count
        for i in range(0, n, 2 ** (qubit + 1)):
            for j in range(2**qubit):
                idx1 = i + j
                idx2 = i + j + 2**qubit
                self.state_vector[idx1], self.state_vector[idx2] = np.dot(
                    gate, [self.state_vector[idx1], self.state_vector[idx2]]
                )

    def apply_cnot(self, control, target):
        n = 2**self.qubits_count
        for i in range(n):
            if (i & (1 << control)) and not (i & (1 << target)):
                j = i ^ (1 << target)
                self.state_vector[i], self.state_vector[j] = (
                    self.state_vector[j],
                    self.state_vector[i],
                )

    def apply_ccnot(self, control1, control2, target):
        n = 2**self.qubits_count
        for i in range(n):
            if (
                (i & (1 << control1))
                and (i & (1 << control2))
                and not (i & (1 << target))
            ):
                j = i ^ (1 << target)
                self.state_vector[i], self.state_vector[j] = (
                    self.state_vector[j],
                    self.state_vector[i],
                )

    def apply_cswap(self, control, target1, target2):
        n = 2**self.qubits_count
        for i in range(n):
            if (i & (1 << control)) and ((i & (1 << target1)) != (i & (1 << target2))):
                j = i ^ (1 << target1) ^ (1 << target2)
                self.state_vector[i], self.state_vector[j] = (
                    self.state_vector[j],
                    self.state_vector[i],
                )

    def apply_swap(self, target1, target2):
        n = 2**self.qubits_count
        for i in range(n):
            #if (i & (1 << target1)) != (i & (1 << target2)):
            j = i ^ (1 << target1) ^ (1 << target2)
            self.state_vector[i], self.state_vector[j] = (
                self.state_vector[j],
                self.state_vector[i],
            )
        return self.state_vector

    def eval_state_vector(self):
        if not self.evaluated:
            self.state_vector[0] = 1  # Initialize to |0...0>
            for gate, qubits in self.circuit:
                qubit = qubits[0]
                if gate == "H":
                    self.apply_sqg(H, qubit)
                elif gate == "X":
                    self.apply_sqg(PX, qubit)
                elif gate == "Y":
                    self.apply_sqg(PY, qubit)
                elif gate == "Z":
                    self.apply_sqg(PZ, qubit)

                elif gate == "P":
                    theta = qubits[1]
                    PHASE = np.array([[1, 0], [0, np.exp(1j * theta)]])
                    self.apply_sqg(PHASE, qubit)

                elif gate == "RX":
                    theta = qubits[1]
                    RX = np.array([
                        [np.cos(theta / 2.0), -1j * np.sin(theta / 2.0)],
                        [-1j * np.sin(theta / 2.0), np.cos(theta / 2)],
                    ])
                    self.apply_sqg(RX, qubit)
                elif gate == "RY":
                    theta = qubits[1]
                    RY = np.array([
                        [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
                        [np.sin(theta / 2.0), np.cos(theta / 2.0)],
                    ])
                    self.apply_sqg(RY, qubit)
                elif gate == "RZ":
                    theta = qubits[1]
                    RZ = np.array([
                        [np.exp(-1j * (theta / 2.0)), 0],
                        [0, np.exp(1j * (theta / 2.0))],
                    ])
                    self.apply_sqg(RZ, qubit)

                elif gate == "CNOT":
                    control, target = qubits
                    self.apply_cnot(control, target)
                elif gate == "SWAP":
                    target1, target2 = qubits
                    self.apply_swap(target1, target2)
                elif gate == "CCNOT":
                    control1, control2, target = qubits
                    self.apply_ccnot(control1, control2, target)
                elif gate == "CSWAP":
                    control, target1, target2 = qubits
                    self.apply_cswap(control, target1, target2)


    def dump(self):
        self.eval_state_vector()
        self.evaluated = True

        probabilities = np.abs(self.state_vector) ** 2
        phases = np.angle(self.state_vector)

        table = [["Basis State", "Probability", "Amplitude", "Phase"]]
        bits = f"{self.qubits_count}"
        for i, prob in enumerate(probabilities):
            if prob > 1e-7:  # Ignore very small probabilities
                sign = "+"
                if self.state_vector[i].imag < 0:
                    sign = "-"
                amplitude = f"{self.state_vector[i].real:.6f} {sign} {abs(self.state_vector[i].imag):.6f}i"

                row = [
                    f"|{i:0{bits}b}>",
                    f"{(prob * 100):.6f}%",
                    f"{amplitude}",
                    f"{phases[i]}",
                ]
                table.append(row)

        print(
            tabulate(
                table[1:], headers=table[0], tablefmt="heavy_grid", stralign="centre"
            )
        )

    def measure_all(self):
        """
        Calculate the probability of measuring the i-th qubit in the |0⟩ and |1⟩ states.

        Args:
            i (int): The index of the qubit.

        Returns:
            Tuple[float, float]: A tuple containing the probabilities (p_zero, p_one).

        Raises:
            IndexError if i < 0 or i > self.qubits_count
        """
        self.eval_state_vector()
        self.evaluated = True
        prob = np.abs(self.state_vector)**2
        prob /= np.sum(prob)
        basis_state = np.random.choice(2**self.qubits_count, p=prob)
        new_state_vector = np.zeros(2**self.qubits_count, dtype=complex)

        new_state_vector[basis_state] = 1.0
        self.state_vector = new_state_vector
        return bin(basis_state)[2:]


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
            "P": "-P",
            "CCNOT": "⨁",
            # Add more gate symbols as needed
        }

        num_qubits = self.qubits_count
        num_gates = len(circuit)
        if header != "":
            print(header)
        for qubit in range(num_qubits):
            theta_gates = -1
            theta_len = 0
            entangle = [
                " "
                for _ in range(
                    3 * num_gates + 4 + len(self.thetas) + 2 * len(self.thetas)
                )
            ]
            line_str = ""
            line_str += f"|q{qubit}⟩"
            for gate_index in range(num_gates):
                gate, targets = circuit[gate_index]
                TARGET = targets[-1]
                if gate in ("Rx", "Ry", "Rz", "P"):
                    theta_gates += 1
                    theta_len += len(str(self.thetas[theta_gates]))

                if qubit in targets:
                    if len(targets) > 1:
                        if gate == "SWAP":
                            line_str += "—x—"
                        elif qubit == targets[0]:
                            line_str += "—●—"
                        elif qubit == TARGET:
                            line_str += f"—{gate_symbols[gate]}—"
                        elif gate == "CSWAP":
                            line_str += "—x—"
                        else:
                            line_str += "—●—"

                        if (qubit < TARGET and qubit >= targets[0]) or (
                            qubit >= TARGET and qubit < targets[0]
                        ):
                            entangle[3 * gate_index + 5 + 8 * (theta_gates + 1)] = "│"

                        if (len(targets) == 3) and (
                            (qubit < TARGET and qubit >= targets[1])
                            or (qubit >= TARGET and qubit < targets[1])
                        ):
                            entangle[3 * gate_index + 5 + 8 * (theta_gates + 1)] = "│"

                    else:
                        if gate in ("Rx", "Ry", "Rz", "P"):
                            line_str += (
                                f"—{gate_symbols[gate]}({self.thetas[theta_gates]})"
                            )
                            # rxyz take a variable argument theta. Pxyz always rotate by π

                        else:
                            line_str += f"—{gate_symbols[gate]}—"

                else:
                    if gate in ("Rx", "Ry", "Rz", "P"):
                        line_str += "—" * 11
                    else:
                        if qubit < max(targets) and qubit > min(targets):
                            line_str += "—│—"
                            entangle[3 * gate_index + 5 + 8 * (theta_gates + 1)] = "│"
                        else:
                            line_str += "—" * 3
            print(line_str)
            print("".join(entangle))


qc = QuantumCircuit(10)
qc.h(0)
qc.cnot(0, 1)
qc.cnot(1, 2)
qc.cnot(2, 3)
qc.cnot(3, 4)
qc.cnot(4, 5)
qc.cnot(5, 6)
qc.cnot(6, 7)
qc.cnot(7, 8)
qc.cnot(8, 9)
"""can't use measure_all before dump, fucks up the statevector"""
"""because i haven't collapsed it yet"""

qc.dump()
print("measured value: ", qc.measure_all())

