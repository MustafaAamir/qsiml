from typing import List, Tuple
from tabulate import tabulate
import numpy as np

COMPLEX_ZERO = complex(0)
COMPLEX_ONE = complex(1)
INITIAL_STATE = [COMPLEX_ONE, COMPLEX_ZERO]
ZERO_STATE = INITIAL_STATE
ONE_STATE = [COMPLEX_ZERO, COMPLEX_ONE]


def _check_index(i, qubits_count: int):
    if i < 0 or i > qubits_count - 1:
        raise IndexError(
            f"Qubit index '{i}' is out of range. Valid range is 0 to {qubits_count - 1}"
        )


def _check_distinct(args: List[int]):
    if len(args) != len(set(args)):
        raise ValueError("Arguments to need to be distinct")

def _check_n(n: int):
    if n > 30:
        raise ValueError(
                f"Qsiml supports circuits with <31 qubits. {n} > 30."
                )


class Qubit:
    def __init__(self):
        """
        Initializes a new Qubit object with an initial state of [1, 0].
        """
        self.states: List[complex] = INITIAL_STATE


class ClassicalBit:
    def __init__(self, value=None):
        self.bit: int | None = value


class QuantumCircuit:
    """
    Initializes a new QuantumCircuit object with the given number of qubits, n.

    Args:
        n (int, optional): The number of qubits in the circuit. Defaults to 1.
    """

    def __init__(self, n: int = 1):
        _check_n(n)
        self.qubits: List[Qubit] = [Qubit() for _ in range(n)]
        # Stores the collapsed state of the nth qubit at index n
        self.classical_bits: List[int | None] = [ClassicalBit().bit for _ in range(n)]
        self.qubits_count: int = n
        self.__thetas: List[str] = []
        # measures contains measured values
        self.__measures: List[int] = []
        self.len_of_thetas = 0
        self.__measures_in: List[int] = []
        self.circuit: List[Tuple[str, List[int | float]]] = []
        self.state_vector = np.zeros(2**self.qubits_count, dtype=complex)
        self.__evaluated: bool = False

    def px(self, i: int):
        """
        Apply the Pauli-X gate (NOT gate) to the i-th qubit.
        The Pauli-X gate flips the state of the qubit, transforming |0⟩ to |1⟩ and vice versa.

        Args:
            i (int): The index of the qubit to apply the gate to.

        Raises:
            IndexError if the index 'i' is out of range
        """
        _check_index(i, self.qubits_count)
        self.circuit.append(("X", [i]))
        self.__evaluated = False

    def py(self, i: int):
        """
        Apply the Pauli-Y gate to the i-th qubit.
        The Pauli-Y gate rotates the qubit state around the Y-axis of the Bloch sphere by π radians.

        Args:
            i (int): The index of the qubit to apply the gate to.

        Raises:
            IndexError if the index 'i' is out of range
        """
        _check_index(i, self.qubits_count)
        self.circuit.append(("Y", [i]))
        self.__evaluated = False

    def pz(self, i: int):
        """
        Apply the Pauli-Z gate to the i-th qubit.
        The Pauli-Z gate rotates the qubit state around the Z-axis of the Bloch sphere by π radians.

        Args:
            i (int): The index of the qubit to apply the gate to.

        Raises:
            IndexError if the index 'i' is out of range
        """

        _check_index(i, self.qubits_count)
        self.circuit.append(("Z", [i]))
        self.__evaluated = False

    def theta_que(self, theta):
        if theta == int(theta):
            theta_len = len(str(theta))
            theta = str(int(theta)) + "." + "0" * (7 - theta_len)
        elif len(str(theta)) < 6:
            theta_len = len(str(float(theta)))
            theta = str(float(theta)) + "0" * (6 - theta_len)
        else:
            theta = str(float(theta))[:6]
        self.len_of_thetas += len((theta))
        self.__thetas.append(str(theta))

    # Rotation Gates
    def rx(self, i: int, theta: float):
        """
        Apply the rx(θ) to the i-th qubit.
        The rx gate rotates the qubit state around the X-axis of the Bloch sphere by theta radians.

        Args:
            i (int): The index of the qubit to apply the gate to.
            theta (float): The angle of rotation in radians.

        Raises:
            IndexError if the index 'i' is out of range
        """

        _check_index(i, self.qubits_count)
        self.circuit.append(("Rx", [i]))
        self.theta_que(float(theta))
        self.__evaluated = False

    def ry(self, i: int, theta: float):
        """
        Apply the rx(θ) to the i-th qubit.
        The ry gate rotates the qubit state around the Y-axis of the Bloch sphere by theta radians.
        Args:
            i (int): The index of the qubit to apply the gate to.
            theta (float): The angle of rotation in radians.

        Raises:
            IndexError if the index 'i' is out of range
        """

        _check_index(i, self.qubits_count)
        self.circuit.append(("Ry", [i, float(theta)]))
        self.theta_que(float(theta))
        self.__evaluated = False

    def rz(self, i: int, theta: float):
        """
        Apply the rz(θ) to the i-th qubit.
        The rz gate rotates the qubit state around the Z-axis of the Bloch sphere by theta radians.
        Args:
            i (int): The index of the qubit to apply the gate to.
            theta (float): The angle of rotation in radians.

        Raises:
            IndexError if the index 'i' is out of range
        """
        _check_index(i, self.qubits_count)
        self.circuit.append(("Rz", [i, float(theta)]))
        self.theta_que(float(theta))
        self.__evaluated = False

    def phase(self, i: int, theta: float):
        """
        Apply a phase shift to the i-th qubit.
        phase(θ) adds a phase e^(i*θ) to the |1⟩, leaving |0⟩ unchanged.

        Args:
            i (int): The index of the qubit to apply the gate to.
            theta (float): The phase angle in radians.

        Raises:
            IndexError if the index 'i' is out of range
        """

        _check_index(i, self.qubits_count)
        self.circuit.append(("P", [i, float(theta)]))
        self.theta_que(float(theta))
        self.__evaluated = False

    def swap(self, i: int, j: int):
        """
        Swap the states of two qubits.

        Args:
            i (int): The index of the first qubit.
            j (int): The index of the second qubit.

        Raises:
            ValueError if the indices 'i', 'j' aren't distinct
        """
        _check_index(i, self.qubits_count)
        _check_index(j, self.qubits_count)
        _check_distinct([i, j])
        self.circuit.append(("SWAP", [i, j]))
        self.__evaluated = False

    def cnot(self, i: int, j: int):
        """
        Apply the CNOT gate with qubit i as control and qubit j as target.
        If the control qubit is |1⟩, the amplitudes of the target qubit are flipped.

        Args:
            i (int): The index of the control qubit.
            j (int): The index of the target qubit.

        Raises:
            ValueError if the indices 'i', 'j' aren't distinct
        """
        _check_distinct([i, j])
        _check_index(i, self.qubits_count)
        _check_index(j, self.qubits_count)
        self.circuit.append(("CNOT", [i, j]))
        self.__evaluated = False

    def h(self, i: int):
        """
        Apply the Hadamard (H) gate to the i-th qubit.
        The H gate creates an equal superposition of |0⟩ and |1⟩ states.
        Args:
            i (int): The index of the qubit to apply the gate to.

        Raises:
            IndexError if the index 'i' is out of range
        """
        _check_index(i, self.qubits_count)
        self.circuit.append(("H", [i]))
        self.__evaluated = False

    def i(self, i: int):
        """
        Apply the Identity (I) gate to the i-th qubit.

        Args:
            i (int): The index of the qubit to apply the gate to.

        Raises:
            IndexError if the index 'i' is out of range
        """
        _check_index(i, self.qubits_count)
        self.circuit.append(("I", [i]))
        self.__evaluated = False

    def cswap(self, i: int, j: int, k: int):
        """
        Apply a CSWAP (Fredkin) gate.
        Swaps the amplitudes of qubits j and k if qubit i is in the |1⟩ state.

        Args:
            i (int): The index of the control qubit.
            j (int): The index of the first target qubit.
            k (int): The index of the second target qubit.

        Raises:
            IndexError if the indices 'i', 'j', 'k' are out of range
            ValueError if the indices 'i', 'j', 'k' aren't distinct
        """
        _check_distinct([i, j, k])
        _check_index(i, self.qubits_count)
        _check_index(j, self.qubits_count)
        _check_index(k, self.qubits_count)
        self.circuit.append(("CSWAP", [i, j, k]))
        self.__evaluated = False

    def ccnot(self, i: int, j: int, k: int):
        """
        Apply a CCNOT (Toffoli) gate.
        Swaps the amplitudes of qubit k if both qubits i and j are in the |1⟩ state.

        Args:
            i (int): The index of the first control qubit.
            j (int): The index of the second control qubit.
            k (int): The index of the target qubit.


        Raises:
            IndexError if the indices 'i', 'j', 'k' are out of range
            ValueError if the indices 'i', 'j', 'k' aren't distinct
        """
        _check_index(i, self.qubits_count)
        _check_index(j, self.qubits_count)
        _check_index(k, self.qubits_count)
        _check_distinct([i, j, k])
        self.circuit.append(("CCNOT", [i, j, k]))
        self.__evaluated = False

    def reset(self, i):
        """
        Reset the i-th qubit to the |0⟩ state.

        Args:
            i (int): The index of the qubit to reset.

        Raises:
            IndexError if the index 'i' is out of range
        """

        _check_index(i, self.qubits_count)
        self.qubits[i].states = INITIAL_STATE
        self.__evaluated = False

    def reset_all(self):
        """
        Reset every qubit in the circuit to the |0⟩ state.
        """
        self.qubits: List[Qubit] = [Qubit() for _ in range(self.qubits_count)]
        self.__thetas: List[str] = []
        self.circuit: List[Tuple[str, List[int | float]]] = []
        self.state_vector = np.zeros(2**self.qubits_count, dtype=complex)
        self.__evaluated = False

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
        """
        Prints the quantum circuit in a format suitable for copy-pasting into Q# katas.

        Args:
            header (str, optional): A header string to print at the top of the output.
        """
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
        print("\t\tDumpMachine();")
        print("\t\tResetAll(qs);")

    def __apply_sqg(self, gate, qubit):
        """
        Applies a single-qubit gate to a specified qubit in the quantum circuit.

        Args:
            gate (np.ndarray): A 2x2 numpy array representing the single-qubit gate to be applied.
            qubit (int): The index of the qubit to apply the gate to.

        Raises:
            IndexError: If the specified qubit index is out of range.
        """
        n = 2**self.qubits_count
        for i in range(0, n, 2 ** (qubit + 1)):
            for j in range(2**qubit):
                idx1 = i + j
                idx2 = i + j + 2**qubit
                self.state_vector[idx1], self.state_vector[idx2] = np.dot(
                    gate, [self.state_vector[idx1], self.state_vector[idx2]]
                )

    def __apply_cnot(self, control, target):
        """
        Applies a controlled-NOT gate to the control and target qubits.

        Args:
            control (int): The index of the control qubit.
            target (int): The index of the target qubit.

        Raises:
            IndexError if either index is out of range.
            ValueError if the indices are not distinct.
        """
        n = 2**self.qubits_count
        for i in range(n):
            if (i & (1 << control)) and not i & (1 << target):
                j = i ^ (1 << target)
                self.state_vector[i], self.state_vector[j] = (
                    self.state_vector[j],
                    self.state_vector[i],
                )

    def __apply_ccnot(self, control1, control2, target):
        """
        Applies a controlled-CNOT gate to the control and target qubits.

        Args:
            control1 (int): The index of the first control qubit.
            control2 (int): The index of the second control qubit.
            target (int): The index of the target qubit.

        Raises:
            IndexError if any index is out of range.
            ValueError if the indices are not distinct.
        """
        n = 2**self.qubits_count
        for i in range(n):
            if (
                (i & (1 << control1))
                and (i & (1 << control2))
                and not i & (1 << target)
            ):
                j = i ^ (1 << target)
                self.state_vector[i], self.state_vector[j] = (
                    self.state_vector[j],
                    self.state_vector[i],
                )

    def __apply_cswap(self, control, target1, target2):
        """
        Applies a controlled-SWAP gate to the two target qubits, controlled by the control qubit.

        Args:
            control (int): The index of the control qubit.
            target1 (int): The index of the first target qubit.
            target2 (int): The index of the second target qubit.

        Raises:
            IndexError if any index is out of range.
            ValueError if any pair of indices are not distinct.
        """
        n = 2**self.qubits_count
        nsv = np.zeros(n, dtype=complex)
        for i in range(n):
            if i & (1 << control):
                idx1 = (i >> target1) & 1
                idx2 = (i >> target2) & 1
                if idx1 != idx2:
                    idx_new = i ^ (1 << target1) ^ (1 << target2)
                else:
                    idx_new = i
            else:
                idx_new = i
            nsv[idx_new] = self.state_vector[i]

        self.state_vector = nsv

    def __apply_swap(self, target1, target2):
        """
        Applies a SWAP gate to the two given qubits.

        Args:
            target1 (int): The index of the first qubit.
            target2 (int): The index of the second qubit.

        Raises:
            IndexError if either index is out of range.
            ValueError if the indices are not distinct.
        """
        n = 2**self.qubits_count
        nsv = np.zeros(n, dtype=complex)
        for i in range(n):
            idx1 = (i >> target1) & 1
            idx2 = (i >> target2) & 1
            if idx1 != idx2:
                idx_new = i ^ (1 << target1) ^ (1 << target2)
            else:
                idx_new = i
            nsv[idx_new] = self.state_vector[i]

        self.state_vector = nsv

    def __apply_measure(self, qubit: int):
        pone = 0
        ret = 0
        for i in range(2**self.qubits_count):
            if i & (1 << qubit):
                pone += np.abs(self.state_vector[i]) ** 2
        if np.random.random() < pone:
            ret = 1

        nsv = np.zeros(2**self.qubits_count, dtype=complex)
        for i in range(2**self.qubits_count):
            if (i & (1 << qubit)) == (ret << qubit):
                nsv[i] = self.state_vector[i]

        norm = np.linalg.norm(nsv)
        if norm != 0:
            nsv /= norm

        self.state_vector = nsv

        self.__measures.append(ret)
        self.classical_bits[qubit] = ret

    def _eval_state_vector(self):
        """
        Evaluates the state vector of the quantum circuit.
        """

        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        PX = np.array([[0, 1], [1, 0]])

        PY = np.array([[0, -1j], [1j, 0]])

        PZ = np.array([[1, 0], [0, -1]])

        ID = np.array([[1, 0], [0, 1]])

        if not self.__evaluated:
            self.state_vector[0] = 1  # Initialize to |0...0>
            for gate, qubits in self.circuit:
                qubit = int(qubits[0])
                if gate == "H":
                    self.__apply_sqg(H, qubit)
                elif gate == "X":
                    self.__apply_sqg(PX, qubit)
                elif gate == "Y":
                    self.__apply_sqg(PY, qubit)
                elif gate == "Z":
                    self.__apply_sqg(PZ, qubit)
                elif gate == "I":
                    self.__apply_sqg(ID, qubit)
                elif gate == "M":
                    self.__apply_measure(qubit)

                elif gate == "P":
                    theta = qubits[1]
                    PHASE = np.array([[1, 0], [0, np.exp(1j * theta)]])
                    self.__apply_sqg(PHASE, qubit)

                elif gate == "RX":
                    theta = qubits[1]
                    RX = np.array([
                        [np.cos(theta / 2.0), -1j * np.sin(theta / 2.0)],
                        [-1j * np.sin(theta / 2.0), np.cos(theta / 2)],
                    ])
                    self.__apply_sqg(RX, qubit)
                elif gate == "RY":
                    theta = qubits[1]
                    RY = np.array([
                        [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
                        [np.sin(theta / 2.0), np.cos(theta / 2.0)],
                    ])
                    self.__apply_sqg(RY, qubit)
                elif gate == "RZ":
                    theta = qubits[1]
                    RZ = np.array([
                        [np.exp(-1j * (theta / 2.0)), 0],
                        [0, np.exp(1j * (theta / 2.0))],
                    ])
                    self.__apply_sqg(RZ, qubit)

                elif gate == "CNOT":
                    control, target = qubits
                    self.__apply_cnot(control, target)
                elif gate == "SWAP":
                    target1, target2 = qubits
                    self.__apply_swap(target1, target2)
                elif gate == "CCNOT":
                    control1, control2, target = qubits
                    self.__apply_ccnot(control1, control2, target)
                elif gate == "CSWAP":
                    control, target1, target2 = qubits
                    self.__apply_cswap(control, target1, target2)

    def dump(self, msg: str = "", format_: str = "outline"):
        """
        Prints all the basis states of the quantum circuit in a human-readable format.

        Args:
            msg (str, optional): A header string to print at the top of the output.
            format_ (str, optional): The output format of the table ("plain" by default)

        Available formats:
            1.   "plain",
            2.   "simple",
            3.   "github",
            4.   "grid",
            5.   "simple_grid",
            6.   "rounded_grid",
            7.   "heavy_grid",
            8.   "mixed_grid",
            9.   "double_grid",
            10.  "fancy_grid",
            11.  "outline",
            12.  "simple_outline",
            13.  "rounded_outline",
            14.  "heavy_outline",
            15.  "mixed_outline",
            16.  "double_outline",
            17.  "fancy_outline",
            18.  "pipe",
            19.  "orgtbl",
            20.  "asciidoc",
            21.  "jira",
            22.  "presto",
            23.  "pretty",
            24.  "psql",
            25.  "rst",
            26.  "mediawiki",
            27.  "moinmoin",
            28.  "youtrack",
            29.  "html",
            30.  "unsafehtml",
            31.  "latex",
            32.  "latex_raw",
            33.  "latex_booktabs",
            34.  "latex_longtable",
            35.  "textile",
            36.  "tsv"

        Raises:
            ValueError: if 'format' is not in the list of available formats
        """
        formats = [
            "plain",
            "simple",
            "github",
            "grid",
            "simple_grid",
            "rounded_grid",
            "heavy_grid",
            "mixed_grid",
            "double_grid",
            "fancy_grid",
            "outline",
            "simple_outline",
            "rounded_outline",
            "heavy_outline",
            "mixed_outline",
            "double_outline",
            "fancy_outline",
            "pipe",
            "orgtbl",
            "asciidoc",
            "jira",
            "presto",
            "pretty",
            "psql",
            "rst",
            "mediawiki",
            "moinmoin",
            "youtrack",
            "html",
            "unsafehtml",
            "latex",
            "latex_raw",
            "latex_booktabs",
            "latex_longtable",
            "textile",
            "tsv",
        ]
        self._eval_state_vector()
        self.__evaluated = True

        print(msg)
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
                    f"|{i:0{bits}b}⟩",
                    f"{(prob * 100):.6f}%",
                    f"{amplitude}",
                    f"{phases[i]}",
                ]
                table.append(row)
        if format_ not in formats:
            raise ValueError(
                f'"{format_}" is not in the available list of formats. For more information, consult the documentation on pypi'
            )

        print(tabulate(table[1:], headers=table[0], tablefmt=format_))

    def measure_all(self):
        """
        measures all qubits and their respective states. Collapses the state vector to one of the basis states.

        Returns:
            str: the basis state it collapses to
        """
        self._eval_state_vector()
        self.__evaluated = True
        prob = np.abs(self.state_vector) ** 2
        prob /= np.sum(prob)
        basis_state = np.random.choice(2**self.qubits_count, p=prob)
        new_state_vector = np.zeros(2**self.qubits_count, dtype=complex)

        new_state_vector[basis_state] = 1.0
        self.state_vector = new_state_vector
        for i, bit in enumerate(bin(basis_state)[2:]):
            self.classical_bits[i] = int(bit)
        return bin(basis_state)[2:]

    def measure(self, i: int):
        _check_index(i, self.qubits_count)
        self.circuit.append(("M", [i]))
        self.__measures_in.append(i)

    def draw(self, header: str = ""):
        """
        Print an ASCII representation of the quantum circuit.

        Args:
            header (str, optional): An optional header to print above the circuit representation. Defaults to "".
        """
        circuit = self.circuit

        GATE_SYMBOLS = {
            "H": "H",
            "I": "I",
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
        }
        if self.__measures == []:
            self._eval_state_vector()
        num_qubits = self.qubits_count
        num_gates = len(circuit)
        if num_qubits < 11:
            padding = 1
        elif num_qubits < 101:
            padding = 2
        elif num_qubits < 1001:
            padding = 3
        elif num_qubits < 10001:
            padding = 4
        elif num_qubits < 100001:
            padding = 5
        else:
            padding = 6

        if header != "":
            print(header)

        for qubit in range(num_qubits):
            theta_gates = -1
            theta_len = 0
            entangle = [
                " "
                for _ in range(
                    3 * num_gates
                    + 3
                    + self.len_of_thetas
                    + 2 * len(self.__thetas)
                    + padding
                )
            ]
            line_str = ""
            padding_str = f"{{:0{padding}d}}"
            qubit_display = f"{padding_str}".format(qubit)
            line_str += f"|q{qubit_display}⟩"
            for gate_index in range(num_gates):
                gate, targets = circuit[gate_index]
                TARGET = targets[-1]
                if gate in ("Rx", "Ry", "Rz", "P"):
                    theta_gates += 1
                    theta_len += len(str(self.__thetas[theta_gates]))

                if qubit in targets and not (
                    qubit == TARGET and gate in ("Rx", "Ry", "Rz", "P")
                ):
                    if len(targets) > 1 and not (isinstance(targets[1], float)):
                        if gate == "SWAP":
                            line_str += "—x—"
                        elif qubit == targets[0]:
                            line_str += "—●—"
                        elif qubit == TARGET:
                            line_str += f"—{GATE_SYMBOLS[gate]}—"
                        elif gate == "CSWAP":
                            line_str += "—x—"
                        else:
                            line_str += "—●—"

                        if (qubit < TARGET and qubit >= targets[0]) or (
                            qubit >= TARGET and qubit < targets[0]
                        ):
                            entangle[
                                (padding - 1)
                                + 3 * gate_index
                                + 5
                                + 8 * (theta_gates + 1)
                            ] = "│"

                        if (len(targets) == 3) and (
                            (qubit < TARGET and qubit >= targets[1])
                            or (qubit >= TARGET and qubit < targets[1])
                        ):
                            entangle[
                                (padding - 1)
                                + 3 * gate_index
                                + 5
                                + 8 * (theta_gates + 1)
                            ] = "│"

                    else:
                        if gate in ("Rx", "Ry", "Rz", "P"):
                            line_str += (
                                f"—{GATE_SYMBOLS[gate]}({self.__thetas[theta_gates]})"
                            )
                        else:
                            if gate in ("M"):
                                entangle[
                                    (padding - 1)
                                    + 3 * gate_index
                                    + 5
                                    + 8 * (theta_gates + 1)
                                ] = str(
                                    self.__measures[self.__measures_in.index(qubit)]
                                )
                            line_str += f"—{GATE_SYMBOLS[gate]}—"

                else:
                    if gate in ("Rx", "Ry", "Rz", "P"):
                        line_str += "—" * 11
                    else:
                        if qubit < max(targets) and qubit > min(targets):
                            line_str += "—│—"
                            entangle[
                                (padding - 1)
                                + 3 * gate_index
                                + 5
                                + 8 * (theta_gates + 1)
                            ] = "│"
                        else:
                            line_str += "—" * 3
            print(line_str)
            print("".join(entangle))


class DeutschJozsa:
    def __init__(self, n: int = 10):
        self.qc = QuantumCircuit(n + 1)
        self.n = n

    def __constant_oracle(self, constant_value: int):
        if constant_value == 0:
            self.qc.i(self.n)
        else:
            self.qc.px(self.n)

    def __balanced_oracle(self, random_bits: int):
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
            self.__constant_oracle(constant_value)
        else:
            self.__balanced_oracle(random_bits)

        for i in range(n):
            self.qc.h(i)

        for i in range(n):
            self.qc.measure(i)

        self.qc.draw()
        print("Classical Bits: ", self.qc.classical_bits[:-1])


qc = QuantumCircuit(31)
