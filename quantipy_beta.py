import random
import cmath
from typing import List, Tuple
import tabulate

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


class Qubit:
    def __init__(self):
        self.states: List[complex] = INITIAL_STATE
        self.dependent = False
        self.entanglements: List[List[Qubit]] = []
        self.gates: List[str] = []
        self.measurement = None
        self.swap = False
        self.measured = False

    def probability(self):
        pzero = abs(self.states[0]) ** 2
        random_float = random.random()
        if random_float < pzero:
            ret = 0
            self.states = ZERO_STATE
        else:
            ret = 1
            self.states = ONE_STATE
        return ret

    def measure(self):
        if self.measured:
            return self.measurement
        else:
            self.measured = True
            if not self.dependent:
                ret = self.probability()
                self.measurement = ret
                return ret
            else:
                cache = {}
                for i in range(len(self.entanglements)):
                    entangled_measurements = []

                    for j in range(len(self.entanglements[i])):
                        if j not in cache:
                            cache[j] = self.entanglements[i][j].measure()
                        entangled_measurements.append([
                            cache[j],
                            self.entanglements[i][j],
                        ])

                    self.measurement = self.apply_gates(
                        self.gates[i], entangled_measurements
                    )
                    return self.measurement

    def apply_gates(self, gate, control):
        if control[0] is None:
            return self.probability()
        if gate == "cnot":
            if control[0][0] == 1:
                self.states = [self.states[1], self.states[0]]
            return self.probability()

        if gate == "cswap":
            if control[0][0] == 1:
                if self.swap:
                    self.states = control[1][1].prev
                else:
                    self.prev = self.states
                    self.states = control[1][1].states
                    control[1][1].states = self.prev

            return self.probability()
        if gate == "ccnot":
            if control[0][0] == 1 and control[1][0] == 1:
                self.states = [self.states[1], self.states[0]]
            return self.probability()

            # code comes here we need to figure this out


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
        self.qubits: List[Qubit] = [Qubit() for _ in range(n)]
        self.thetas: List[float] = []
        self.Len = 0
        self.qubits_count: int = n
        self.circuit: List[Tuple[str, List[int]]] = []

    def __repr__(self):
        """
        Return a string representation of every qubit's amplitude in the following format:
            Qubit {i}: [alpha, beta]
        where alpha and beta are the individual probability amplitudes for each state.
        """
        print("Circuit Diagram: ")
        self.draw()
        return ""
        # PaulliGates

    def add_entanglement(self, target, control_bits, gate, swap=False):
        self.qubits[target].entanglements.append(control_bits)
        self.qubits[target].gates.append(gate)
        self.qubits[target].dependent = True
        self.qubits[target].swap = swap

    def theta_que(self, theta):
        if len(str(theta)) > 6:
            theta = float(str(theta)[:6])
        self.Len += len(str(theta))
        self.thetas.append(theta)

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
        self.qubits[i].states = [self.qubits[i].states[1], self.qubits[i].states[0]]
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
        self.qubits[i].states = [
            self.qubits[i].states[1] * -1j,
            self.qubits[i].states[0] * 1j,
        ]
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
        self.qubits[i].states = [
            self.qubits[i].states[0],
            self.qubits[i].states[1] * -1,
        ]
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
        self.qubits[i].states = [
            self.qubits[i].states[0] * cmath.cos(theta / 2)
            + self.qubits[i].states[1] * cmath.sin(theta / 2) * -1j,
            self.qubits[i].states[1] * cmath.cos(theta / 2)
            + self.qubits[i].states[0] * cmath.sin(theta / 2) * -1j,
        ]
        self.circuit.append(("Rx", [i]))
        self.theta_que(theta)

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
        self.qubits[i].states = [
            self.qubits[i].states[0] * cmath.cos(theta / 2)
            + self.qubits[i].states[1] * cmath.sin(theta / 2) * -1,
            self.qubits[i].states[1] * cmath.cos(theta / 2)
            + self.qubits[i].states[0] * cmath.sin(theta / 2),
        ]
        self.circuit.append(("Ry", [i]))
        self.theta_que(theta)

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
        self.qubits[i].states = [
            self.qubits[i].states[0] * cmath.exp(-1j * (theta / 2)),
            self.qubits[i].states[1] * cmath.exp(1j * (theta / 2)),
        ]
        self.circuit.append(("Rz", [i]))
        self.theta_que(theta)

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
        self.qubits[i].states = [
            self.qubits[i].states[0],
            complex(self.qubits[i].states[1] * (cmath.exp(1j * theta))),
        ]
        if theta % cmath.pi == 0:
            self.qubits[i].states[1] = self.qubits[i].states[1].real
        elif (theta % (cmath.pi / 2)) == 0:
            self.qubits[i].states[1] = (self.qubits[i].states[1].imag) * 1j
        self.circuit.append(("P", [i]))
        self.theta_que(theta)

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
        temp = self.qubits[i].states
        self.qubits[i].states = self.qubits[j].states
        self.qubits[j].states = temp
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
        if self.qubits[i].states == ONE_STATE:
            self.qubits[j].states = [self.qubits[j].states[1], self.qubits[j].states[0]]
        self.circuit.append(("CNOT", [i, j]))
        self.add_entanglement(j, [self.qubits[i]], "cnot")

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
        self.qubits[i].states = [
            (1 / (2**0.5)) * (self.qubits[i].states[1] + self.qubits[i].states[0]),
            (1 / (2**0.5))
            * ((-1 * self.qubits[i].states[1]) + self.qubits[i].states[0]),
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
        if self.qubits[i].states == ONE_STATE:
            temp = self.qubits[j].states
            self.qubits[j].states = self.qubits[k].states
            self.qubits[k].states = temp

        self.circuit.append(("CSWAP", [i, j, k]))
        self.add_entanglement(k, [self.qubits[i], self.qubits[j]], "cswap")

        self.add_entanglement(j, [self.qubits[i], self.qubits[k]], "cswap", True)

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
        if self.qubits[i].states == ONE_STATE and self.qubits[j].states == ONE_STATE:
            self.qubits[k].states = [self.qubits[k].states[1], self.qubits[k].states[0]]
        self.circuit.append(("CCNOT", [i, j, k]))
        self.add_entanglement(k, [self.qubits[i], self.qubits[j]], "ccnot")

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
        pzero = abs(self.qubits[i].states[0]) ** 2
        pone = abs(self.qubits[i].states[1]) ** 2
        tp = pzero + pone
        # division by zero errors somehow
        if tp != 0:
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
            self.qubits[i].states = ZERO_STATE
        else:
            ret = 1
            self.qubits[i].states = ONE_STATE
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
        for i in range(self.qubits_count):
            measured_values.append(self.qubits[i].measure())

        return measured_values

    # making chagnes

    def get_nsv(self, msg: str = ""):
        state_vector = [1]
        for qubit in self.qubits:
            nstate_vector = []
            for amp in state_vector:
                # add some kind of permutation
                nstate_vector.append(amp * qubit.states[0])
                nstate_vector.append(amp * qubit.states[1])
            state_vector = nstate_vector

        normalize = 0
        for amp in state_vector:
            normalize += abs(amp) ** 2
        normalize = cmath.sqrt(normalize)
        normalized_state_vector = []

        # if only imaginary amplitudes, prints an empty table
        # BUG: FIXED
        if normalize.real > 0:
            for amp in state_vector:
                normalized_state_vector.append(amp / normalize)
        else:
            for amp in state_vector:
                normalized_state_vector.append(amp)

        return normalized_state_vector

    def dump(self, msg: str = ""):
        """
        Print the current state of the quantum circuit without affecting it.
        Displays the probability amplitudes for each basis state along with their probabilities and phases.

        Args:
            msg (str, optional): An optional message to print before the state dump. Defaults to "".
        """
        print(msg)
        normalized_state_vector = self.get_nsv()
        table = [
            ["Basis state", "Probabilty", "Phase", "Amplitude"],
        ]
        # iterate over
        for i, amp in enumerate(normalized_state_vector):
            prob = abs(amp) ** 2
            row = []
            if prob < 1e-8:
                continue

            basis_state = "|" + format(i, f"{self.qubits_count}b") + "⟩"
            row.append(basis_state)
            row.append("{:.2f}%".format(prob * 100))
            phase = cmath.phase(amp)
            if phase % cmath.pi == 0:
                coefficient = phase // cmath.pi
                phase = str(int(coefficient)) + "π"
                row.append(phase)
            else:
                row.append("{:.4}".format(phase))
            sign = "+"
            if amp.imag < 0:
                sign = "-"
            row.append(
                "{:.4} ".format(amp.real) + sign + " {:.4}i".format(abs(amp.imag))
            )
            table.append(row)
        print(
            tabulate.tabulate(
                table[1:], headers=table[0], tablefmt="heavy_grid", stralign="centre"
            )
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
        self.qubits[i].states = INITIAL_STATE

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
                " " for _ in range(3 * num_gates + 4 + self.Len + 2 * len(self.thetas))
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
        print(self.thetas)
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


# gen random quantum circuits


def gen_rand(n, d):
    """
    n :int => number of desired qubits
    d :int => number of desired gates
    """
    qc = QuantumCircuit(n)
    for _ in range(d):
        random_size = random.randint(0, 2)
        if random_size == 0:
            RANDOM_IDX = random.randint(0, n - 1)
            RANDOM_ANGLE = random.random() * cmath.pi
            # removing rxyz gates for testing
            random_gate = random.choice(["px", "py", "pz", "rx", "ry", "rz", "h", "m"])
            if random_gate == "px":
                qc.px(RANDOM_IDX)
            elif random_gate == "py":
                qc.py(RANDOM_IDX)
            elif random_gate == "pz":
                qc.pz(RANDOM_IDX)
            elif random_gate == "h":
                qc.h(RANDOM_IDX)
            elif random_gate == "m":
                qc.measure(RANDOM_IDX)
            elif random_gate == "rx":
                qc.rx(RANDOM_IDX, RANDOM_ANGLE)
                print(f"applied angle: {RANDOM_ANGLE} to rx")
            elif random_gate == "ry":
                qc.ry(RANDOM_IDX, RANDOM_ANGLE)
                print(f"applied angle: {RANDOM_ANGLE} to ry")
            elif random_gate == "rz":
                qc.rz(RANDOM_IDX, RANDOM_ANGLE)
                print(f"applied angle: {RANDOM_ANGLE} to rz")

        elif random_size == 1:
            random_gate = random.choice(["cnot", "swap"])
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            while b == a:
                b = random.randint(0, n - 1)
            if random_gate == "cnot":
                qc.cnot(a, b)
            else:
                qc.swap(a, b)
        else:
            # removing cswap for testing
            random_gate = random.choice(["ccnot", "cswap"])
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            c = random.randint(0, n - 1)
            while b == a or c == a or c == b:
                b = random.randint(0, n - 1)
                c = random.randint(0, n - 1)
            if random_gate == "ccnot":
                qc.ccnot(a, b, c)
            else:
                qc.cswap(a, b, c)

    return qc


"""
Expected:
    state 110, amp 0-0.8776
    state 111, amp
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃   Basis state ┃   Probabilty ┃   Phase ┃      Amplitude ┃
┣━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━╋━━━━━━━━━╋━━━━━━━━━━━━━━━━┫
┃ |11⟩          ┃ 77.02%       ┃  -1.571 ┃ -0.0 - 0.8776i ┃
┣━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━╋━━━━━━━━━╋━━━━━━━━━━━━━━━━┫
┃ |111⟩         ┃ 22.98%       ┃  -1.571 ┃ -0.0 - 0.4794i ┃
┗━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━┻━━━━━━━━━━━━━━━━┛

"""
"""
qc.ry(8, 2.9616)
qc.rx(4, 1.5976)
qc.ry(0, 0.08473)
qc.swap(4, 0)
qc.swap(0, 9)
qc.ry(0, 0.8611)
qc.rz(1, 0.5157)
"""

"""
qc = QuantumCircuit(10)
qc.px(0)
qc.py(1)
qc.pz(2)
qc.rx(3, 2.5)
qc.ry(4, 2.5)
qc.rz(5, 2.5)
qc.px(6)
qc.py(7)
qc.pz(8)
"""


# qc=QuantumCircuit(2)
# qc.h(0)
# qc.cx(0, 1)
# qc.measure_all()
# expected output:
# 00 or 11

qc = QuantumCircuit(3)
qc.h(0)
qc.px(1)
qc.cswap(0, 1, 2)
print(qc)
print(qc.measure_all())
