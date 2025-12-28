from typing import List, Tuple
from tabulate import tabulate
import numpy as np

INITIAL_STATE = [complex(1), complex(0)]
ZERO_STATE = INITIAL_STATE
ONE_STATE = [complex(0), complex(1)]

# Gate Constants
H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)
I_GATE = np.array([[1, 0], [0, 1]], dtype=complex)
# SWAP and CNOT reshaped for tensordot
# Target indices are (output_control, output_target, input_control, input_target)
SWAP_GATE = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex).reshape(2, 2, 2, 2)
CNOT_GATE = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex).reshape(2, 2, 2, 2)
CCNOT_GATE = np.eye(8, dtype=complex)
CCNOT_GATE[6:, 6:] = [[0, 1], [1, 0]]
CCNOT_GATE = CCNOT_GATE.reshape(2, 2, 2, 2, 2, 2)
CSWAP_GATE = np.eye(8, dtype=complex)
CSWAP_GATE[5, 5] = 0
CSWAP_GATE[5, 6] = 1
CSWAP_GATE[6, 5] = 1
CSWAP_GATE[6, 6] = 0
CSWAP_GATE = CSWAP_GATE.reshape(2, 2, 2, 2, 2, 2)



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
        self.qubits_count = n
        self.classical_bits: List[int | None] = [None] * n
        
        # Initialize state |0...0>
        # Represent state as a tensor of shape (2, 2, ..., 2)
        # Axis i corresponds to qubit (n - 1 - i)
        # i.e., Axis 0 is q(n-1), Axis n-1 is q0.
        self._state_tensor = np.zeros((2,) * n, dtype=complex)
        self._state_tensor[(0,) * n] = 1.0 + 0j
        
        # Legacy/Drawing support
        self.qubits: List[Qubit] = [Qubit() for _ in range(n)] 
        self.circuit: List[Tuple[str, List[int | float]]] = []
        self.__thetas: List[str] = []
        self.__measures: List[int] = []
        self.__measures_in: List[int] = []
        self.len_of_thetas = 0

    @property
    def state_vector(self):
        return self._state_tensor.flatten()
        
    @state_vector.setter
    def state_vector(self, value):
        self._state_tensor = value.reshape((2,) * self.qubits_count)
    
    def _apply_gate(self, gate: np.ndarray, targets: List[int]):
        n = self.qubits_count
        k = len(targets)
        target_axes = [n - 1 - t for t in targets]
        gate_input_axes = list(range(k, 2 * k))
        new_state = np.tensordot(gate, self._state_tensor, axes=(gate_input_axes, target_axes))
        
        target_q_to_current_axis = {q: i for i, q in enumerate(targets)}
        non_target_q_descending = sorted([q for q in range(n) if q not in targets], reverse=True)
        non_target_q_to_current_axis = {q: k + i for i, q in enumerate(non_target_q_descending)}
        
        perm = []
        for j in range(n):
            q = n - 1 - j
            if q in target_q_to_current_axis:
                perm.append(target_q_to_current_axis[q])
            else:
                perm.append(non_target_q_to_current_axis[q])
        self._state_tensor = np.transpose(new_state, perm)

    def measure(self, i: int):
        _check_index(i, self.qubits_count)
        self.circuit.append(("M", [i]))
        self.__measures_in.append(i)
        
        axis = self.qubits_count - 1 - i
        # Calculate prob of 1
        # Move axis to 0
        psi = np.moveaxis(self._state_tensor, axis, 0)
        prob_1 = np.sum(np.abs(psi[1])**2)
        
        outcome = 1 if np.random.random() < prob_1 else 0
        
        # Collapse
        other = 1 - outcome
        idx = [slice(None)] * self.qubits_count
        idx[axis] = other
        self._state_tensor[tuple(idx)] = 0
        
        norm = np.linalg.norm(self._state_tensor)
        if norm > 1e-12:
            self._state_tensor /= norm
        
        self.__measures.append(outcome)
        self.classical_bits[i] = outcome
        return outcome

    def reset(self, i: int):
        _check_index(i, self.qubits_count)
        # Cannot reuse measure easily without appending to circuit log, so simpler logic here?
        # Re-implement collapse logic without logging measure, or just call measure but remove log?
        # The user's code expects reset to be functional.
        
        # Let's peek at the state to decide
        axis = self.qubits_count - 1 - i
        psi = np.moveaxis(self._state_tensor, axis, 0)
        prob_1 = np.sum(np.abs(psi[1])**2)
        outcome = 1 if np.random.random() < prob_1 else 0
        
        # Collapse
        other = 1 - outcome
        idx = [slice(None)] * self.qubits_count
        idx[axis] = other
        self._state_tensor[tuple(idx)] = 0
        
        norm = np.linalg.norm(self._state_tensor)
        if norm > 1e-12:
            self._state_tensor /= norm
            
        if outcome == 1:
            self._apply_gate(X_GATE, [i])
            
        self.qubits[i].states = INITIAL_STATE 

    def measure_all(self):
        probs = np.abs(self.state_vector) ** 2
        probs /= np.sum(probs)
        basis_state = np.random.choice(2**self.qubits_count, p=probs)
        
        # Collapse state
        new_state = np.zeros_like(self.state_vector)
        new_state[basis_state] = 1.0
        self.state_vector = new_state
        
        # Update classical bits
        for i in range(self.qubits_count):
            self.classical_bits[i] = (basis_state >> i) & 1
            
        return bin(basis_state)[2:]


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

    def apply_group(self, gates: List[Tuple[str, List[int | float]]]):
        """
        Apply a group of gates sequentially.
        Since the underlying state update is vectorized and uses BLAS, 
        this effectively leverages CPU parallelization for tensor contractions.
        
        Args:
            gates: List of tuples (gate_name, targets_and_params)
        """
        for gate_name, args in gates:
            gate_func = getattr(self, gate_name.lower())
            if gate_name.lower() in ["rx", "ry", "rz", "phase", "p"]:
                gate_func(int(args[0]), float(args[1]))
            elif gate_name.lower() in ["cnot", "swap"]:
                gate_func(int(args[0]), int(args[1]))
            elif gate_name.lower() in ["ccnot", "cswap"]:
                gate_func(int(args[0]), int(args[1]), int(args[2]))
            else:
                 gate_func(int(args[0]))
    
    def px(self, i: int):
        _check_index(i, self.qubits_count)
        self.circuit.append(("X", [i]))
        self._apply_gate(X_GATE, [i])
        return self

    def py(self, i: int):
        _check_index(i, self.qubits_count)
        self.circuit.append(("Y", [i]))
        self._apply_gate(Y_GATE, [i])
        return self

    def pz(self, i: int):
        _check_index(i, self.qubits_count)
        self.circuit.append(("Z", [i]))
        self._apply_gate(Z_GATE, [i])
        return self

    def h(self, i: int):
        _check_index(i, self.qubits_count)
        self.circuit.append(("H", [i]))
        self._apply_gate(H_GATE, [i])
        return self

    def i(self, i: int):
        _check_index(i, self.qubits_count)
        self.circuit.append(("I", [i]))
        return self

    def rx(self, i: int, theta: float):
        _check_index(i, self.qubits_count)
        self.circuit.append(("Rx", [i]))
        self.theta_que(float(theta))
        theta = float(theta)
        gate = np.array([
            [np.cos(theta / 2.0), -1j * np.sin(theta / 2.0)],
            [-1j * np.sin(theta / 2.0), np.cos(theta / 2)],
        ], dtype=complex)
        self._apply_gate(gate, [i])
        return self

    def ry(self, i: int, theta: float):
        _check_index(i, self.qubits_count)
        self.circuit.append(("Ry", [i, float(theta)]))
        self.theta_que(float(theta))
        theta = float(theta)
        gate = np.array([
             [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
             [np.sin(theta / 2.0), np.cos(theta / 2.0)],
         ], dtype=complex)
        self._apply_gate(gate, [i])
        return self

    def rz(self, i: int, theta: float):
        _check_index(i, self.qubits_count)
        self.circuit.append(("Rz", [i, float(theta)]))
        self.theta_que(float(theta))
        theta = float(theta)
        gate = np.array([
            [np.exp(-1j * (theta / 2.0)), 0],
            [0, np.exp(1j * (theta / 2.0))],
        ], dtype=complex)
        self._apply_gate(gate, [i])
        return self

    def phase(self, i: int, theta: float):
        _check_index(i, self.qubits_count)
        self.circuit.append(("P", [i, float(theta)]))
        self.theta_que(float(theta))
        theta = float(theta)
        gate = np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)
        self._apply_gate(gate, [i])
        return self

    def swap(self, i: int, j: int):
        _check_distinct([i, j])
        _check_index(i, self.qubits_count)
        _check_index(j, self.qubits_count)
        self.circuit.append(("SWAP", [i, j]))
        self._apply_gate(SWAP_GATE, [i, j])
        return self

    def cnot(self, i: int, j: int):
        _check_distinct([i, j])
        _check_index(i, self.qubits_count)
        _check_index(j, self.qubits_count)
        self.circuit.append(("CNOT", [i, j]))
        self._apply_gate(CNOT_GATE, [i, j])
        return self

    def cswap(self, i: int, j: int, k: int):
        _check_distinct([i, j, k])
        _check_index(i, self.qubits_count)
        _check_index(j, self.qubits_count)
        _check_index(k, self.qubits_count)
        self.circuit.append(("CSWAP", [i, j, k]))
        self._apply_gate(CSWAP_GATE, [i, j, k])
        return self

    def ccnot(self, i: int, j: int, k: int):
        _check_distinct([i, j, k])
        _check_index(i, self.qubits_count)
        _check_index(j, self.qubits_count)
        _check_index(k, self.qubits_count)
        self.circuit.append(("CCNOT", [i, j, k]))
        self._apply_gate(CCNOT_GATE, [i, j, k])
        return self

    def reset_all(self):
        self.qubits = [Qubit() for _ in range(self.qubits_count)]
        self.circuit = []
        self.__thetas = []
        self.__measures = []
        self.__measures_in = []
        self.len_of_thetas = 0
        self._state_tensor = np.zeros((2,) * self.qubits_count, dtype=complex)
        self._state_tensor[(0,) * self.qubits_count] = 1.0 + 0j

    def operations(self, header: str = ""):
        print(header)
        for i, (gate, targets) in enumerate(self.circuit):
            qubit_plural = "qubit" if len(targets) > 1 else ""
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
        print("\t\tDumpMachine();")
        print("\t\tResetAll(qs);")

    def dump(self, msg: str = "", format_: str = "outline"):
        formats = [
            "plain", "simple", "github", "grid", "simple_grid", "rounded_grid", "heavy_grid", 
            "mixed_grid", "double_grid", "fancy_grid", "outline", "simple_outline", 
            "rounded_outline", "heavy_outline", "mixed_outline", "double_outline", 
            "fancy_outline", "pipe", "orgtbl", "asciidoc", "jira", "presto", "pretty", 
            "psql", "rst", "mediawiki", "moinmoin", "youtrack", "html", "unsafehtml", 
            "latex", "latex_raw", "latex_booktabs", "latex_longtable", "textile", "tsv"
        ]
        if format_ not in formats:
            raise ValueError(f'"{format_}" is not in the available list of formats.')

        print(msg)
        probabilities = np.abs(self.state_vector) ** 2
        phases = np.angle(self.state_vector)

        table = [["Basis State", "Probability", "Amplitude", "Phase"]]
        bits = f"{self.qubits_count}"
        
        sv_flat = self.state_vector
        for i, prob in enumerate(probabilities):
            if prob > 1e-7:
                sign = "+"
                if sv_flat[i].imag < 0:
                    sign = "-"
                amplitude = f"{sv_flat[i].real:.6f} {sign} {abs(sv_flat[i].imag):.6f}i"
                row = [
                    f"|{i:0{bits}b}⟩",
                    f"{(prob * 100):.6f}%",
                    f"{amplitude}",
                    f"{phases[i]}",
                ]
                table.append(row)

        print(tabulate(table[1:], headers=table[0], tablefmt=format_))

    def draw(self, header: str = ""):
        circuit = self.circuit
        GATE_SYMBOLS = {
            "H": "H", "I": "I", "X": "X", "Y": "Y", "M": "M", "Z": "Z",
            "CNOT": "⨁", "SWAP": "x", "CSWAP": "x",
            "Rx": "Rx", "Ry": "Rʏ", "Rz": "Rᴢ", "P": "-P",
            "CCNOT": "⨁",
        }
        
        num_qubits = self.qubits_count
        num_gates = len(circuit)
        if num_qubits < 11: padding = 1
        elif num_qubits < 101: padding = 2
        elif num_qubits < 1001: padding = 3
        elif num_qubits < 10001: padding = 4
        elif num_qubits < 100001: padding = 5
        else: padding = 6

        if header != "":
            print(header)

        for qubit in range(num_qubits):
            theta_gates = -1
            entangle = [" " for _ in range(3 * num_gates + 3 + self.len_of_thetas + 2 * len(self.__thetas) + padding)]
            line_str = ""
            padding_str = f"{{:0{padding}d}}"
            qubit_display = f"{padding_str}".format(qubit)
            line_str += f"|q{qubit_display}⟩"
            for gate_index in range(num_gates):
                gate, targets = circuit[gate_index]
                TARGET = targets[-1]
                if gate in ("Rx", "Ry", "Rz", "P"):
                    theta_gates += 1

                if qubit in targets and not (qubit == TARGET and gate in ("Rx", "Ry", "Rz", "P")):
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

                        if (qubit < TARGET and qubit >= targets[0]) or (qubit >= TARGET and qubit < targets[0]):
                            entangle[(padding - 1) + 3 * gate_index + 5 + 8 * (theta_gates + 1)] = "│"

                        if (len(targets) == 3) and ((qubit < TARGET and qubit >= targets[1]) or (qubit >= TARGET and qubit < targets[1])):
                             entangle[(padding - 1) + 3 * gate_index + 5 + 8 * (theta_gates + 1)] = "│"
                    else:
                        if gate in ("Rx", "Ry", "Rz", "P"):
                            line_str += f"—{GATE_SYMBOLS[gate]}({self.__thetas[theta_gates]})"
                        else:
                            if gate in ("M"):
                                idx = self.__measures_in.index(qubit) if qubit in self.__measures_in else -1
                                val = str(self.__measures[idx]) if idx < len(self.__measures) and idx != -1 else "?"
                                entangle[(padding - 1) + 3 * gate_index + 5 + 8 * (theta_gates + 1)] = val
                            line_str += f"—{GATE_SYMBOLS[gate]}—"
                else:
                    if gate in ("Rx", "Ry", "Rz", "P"):
                        line_str += "—" * 11
                    else:
                        if qubit < max(targets) and qubit > min(targets):
                            line_str += "—│—"
                            entangle[(padding - 1) + 3 * gate_index + 5 + 8 * (theta_gates + 1)] = "│"
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
