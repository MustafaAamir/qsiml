from typing import List, Tuple
from tabulate import tabulate
import numpy as np

COMPLEX_ZERO = complex(0)
COMPLEX_ONE = complex(1)
INITIAL_STATE = [COMPLEX_ONE, COMPLEX_ZERO]
ZERO_STATE = INITIAL_STATE
ONE_STATE = [COMPLEX_ZERO, COMPLEX_ONE]

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

class VectorBackend:
    def __init__(self, n_qubits):
        self.n = n_qubits
        # State tensor axis i corresponds to qubit (n - 1 - i)
        self.state_tensor = np.zeros((2,) * n_qubits, dtype=complex)
        self.state_tensor[(0,) * n_qubits] = 1.0 + 0j

    @property
    def state_vector(self):
        return self.state_tensor.flatten()
    
    @state_vector.setter
    def state_vector(self, value):
        self.state_tensor = value.reshape((2,) * self.n)

    def apply_gate(self, gate: np.ndarray, targets: List[int]):
        k = len(targets)
        target_axes = [self.n - 1 - t for t in targets]
        gate_input_axes = list(range(k, 2 * k))
        new_state = np.tensordot(gate, self.state_tensor, axes=(gate_input_axes, target_axes))
        
        target_q_to_current_axis = {q: i for i, q in enumerate(targets)}
        non_target_q_descending = sorted([q for q in range(self.n) if q not in targets], reverse=True)
        non_target_q_to_current_axis = {q: k + i for i, q in enumerate(non_target_q_descending)}
        
        perm = []
        for j in range(self.n):
            q = self.n - 1 - j
            if q in target_q_to_current_axis:
                perm.append(target_q_to_current_axis[q])
            else:
                perm.append(non_target_q_to_current_axis[q])
        self.state_tensor = np.transpose(new_state, perm)

    def measure(self, qubit):
        axis = self.n - 1 - qubit
        # Calculate prob of 1
        # Move axis to 0
        psi = np.moveaxis(self.state_tensor, axis, 0)
        prob_1 = np.sum(np.abs(psi[1])**2)
        
        outcome = 1 if np.random.random() < prob_1 else 0
        
        # Collapse
        other = 1 - outcome
        idx = [slice(None)] * self.n
        idx[axis] = other
        self.state_tensor[tuple(idx)] = 0
        
        norm = np.linalg.norm(self.state_tensor)
        if norm > 1e-12:
            self.state_tensor /= norm
            
        return outcome

class MPSBackend:
    def __init__(self, n_qubits, max_bond_dim=32):
        self.n = n_qubits
        self.chi = max_bond_dim
        # Tensors Gamma[i] of shape (chi_left, 2, chi_right)
        # For i=0: (1, 2, chi) [actually (1, 2, 1) initially]
        self.tensors = []
        for i in range(n_qubits):
            # Init to |0> state: [1, 0]
            # Shape (1, 2, 1)
            T = np.zeros((1, 2, 1), dtype=complex)
            T[0, 0, 0] = 1.0
            self.tensors.append(T)

    @property
    def state_vector(self):
        # Contract all tensors to get full vector
        # This is expensive O(2^N), only for debug/small N
        if self.n > 20: 
            raise MemoryError("State vector too large for MPS conversion")
        
        C = self.tensors[0] # (1, 2, chi)
        for i in range(1, self.n):
            # Contract C (..., chi) with T[i] (chi, 2, chi_next)
            # Result (..., 2, chi_next) -> flatten later
            C = np.tensordot(C, self.tensors[i], axes=(-1, 0)) 
            # C shape: (1, 2, ..., 2, 1)
            
        return C.flatten()

    def apply_gate(self, gate: np.ndarray, targets: List[int]):
        if len(targets) == 1:
            self._apply_1q(gate, targets[0])
        elif len(targets) == 2:
            q1, q2 = targets
            if abs(q1 - q2) == 1:
                self._apply_2q_adjacent(gate, min(q1, q2))
            else:
                 # Check if gate is SWAP (special case optimized?)
                 # For now, simplistic SWAP chain or error
                 # But we must support general gates.
                 raise NotImplementedError("Non-adjacent gates not yet supported in MPS (SWAP chain required)")
        else:
            raise NotImplementedError("Multi-qubit gates > 2 not supported in MPS")

    def _apply_1q(self, gate, q):
        # gate shape (2, 2)
        # tensor shape (chi_l, 2, chi_r)
        # Contract gate[out, in] * T[l, in, r] -> R[out, l, r] -> transpose to [l, out, r]
        T = self.tensors[q]
        # tensordot axes: gate(1) with T(1)
        res = np.tensordot(gate, T, axes=(1, 1)) # (out, l, r)
        self.tensors[q] = np.transpose(res, (1, 0, 2))

    def _apply_2q_adjacent(self, gate, q):
        # Apply to q, q+1
        # gate is (2, 2, 2, 2) - [out1, out2, in1, in2] 
        # (Be careful with gate shape conventions from qsiml constants)
        # Standard constants like CNOT_GATE are (out_c, out_t, in_c, in_t) if we view them as matrix ops
        # In VectorBackend we reshaped them. 
        # Let's verify shape: CNOT_GATE was reshaped(2, 2, 2, 2).
        # Target indices are (output_control, output_target, input_control, input_target)
        
        # 1. Contract T[q] and T[q+1]
        # T[q]: (chi_l, 2, chi_mid)
        # T[q+1]: (chi_mid, 2, chi_r)
        # Contract on chi_mid -> theta: (chi_l, 2(q), 2(q+1), chi_r)
        T1 = self.tensors[q]
        T2 = self.tensors[q+1]
        
        theta = np.tensordot(T1, T2, axes=(2, 0)) # (chi_l, 2, 2, chi_r)
        
        # 2. Apply Gate
        # Gate: (out_q, out_qp1, in_q, in_qp1)
        # Contract gate inputs with theta inputs
        # gate axes (2, 3) match theta axes (1, 2)
        theta_prime = np.tensordot(gate, theta, axes=([2, 3], [1, 2])) # (out_q, out_qp1, chi_l, chi_r)
        
        # Transpose to (chi_l, out_q, out_qp1, chi_r)
        theta_prime = np.transpose(theta_prime, (2, 0, 1, 3))
        
        # 3. SVD and Truncate
        # Reshape to Matrix: (chi_l * 2) x (2 * chi_r)
        chi_l = T1.shape[0]
        chi_r = T2.shape[2]
        M = theta_prime.reshape(chi_l * 2, 2 * chi_r)
        
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        
        # Truncate
        rank = len(S)
        keep = min(rank, self.chi)
        # Also trim small singular values?
        keep = min(keep, np.count_nonzero(S > 1e-12))
        if keep < 1: keep = 1
        
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]
        
        # 4. Reshape back
        # U: (chi_l * 2, keep) -> (chi_l, 2, keep)
        # Vh: (keep, 2 * chi_r) -> SVh = diag(S) @ Vh -> (keep, 2 * chi_r) -> (keep, 2, chi_r)
        
        self.tensors[q] = U.reshape(chi_l, 2, keep)
        # Absorb S into V or U. Typically split sqrts or put in one. 
        # Standard: put S into right tensor (canonical form shift) or keep center.
        # Here we put it into right.
        self.tensors[q+1] = (np.diag(S) @ Vh).reshape(keep, 2, chi_r)

    def measure(self, qubit):
        # Calculate Probability of 1: <psi| P1 |psi>
        # We can implement this by temporarily applying P1 to the tensor at 'qubit' and calculating norm squared.
        
        # P1 operator
        P1 = np.array([[0, 0], [0, 1]], dtype=complex)
        P0 = np.array([[1, 0], [0, 0]], dtype=complex)
        
        # Create a copy of the tensor at qubit with P1 applied
        # T[q] shape (chi_l, 2, chi_r) -> contract with P1(out, in)
        # P1 is diagonal, so just zero out the 0-index of middle dimension.
        T_orig = self.tensors[qubit]
        T_1 = np.zeros_like(T_orig)
        T_1[:, 1, :] = T_orig[:, 1, :]
        
        # Calculate norm squared of state with T_1 at qubit
        norm_sq_1 = self._calc_norm_sq(self.tensors[:qubit] + [T_1] + self.tensors[qubit+1:])
        
        prob_1 = norm_sq_1.real # Should be real
        
        outcome = 1 if np.random.random() < prob_1 else 0
        
        # Collapse
        if outcome == 1:
            self.tensors[qubit] = T_1
            current_norm = np.sqrt(norm_sq_1)
        else:
            T_0 = np.zeros_like(T_orig)
            T_0[:, 0, :] = T_orig[:, 0, :]
            self.tensors[qubit] = T_0
            # Need to calc norm 0 to renormalize accurately, or just 1-prob?
            # Better to calc norm 0 from state to be safe against precision errors
            norm_sq_0 = self._calc_norm_sq(self.tensors)
            current_norm = np.sqrt(norm_sq_0)
            
        if current_norm > 1e-12:
            self.tensors[qubit] /= current_norm
            
        return outcome

    def _calc_norm_sq(self, tensors):
        # Contract ladder <psi|psi>
        # Left boundary
        # E (chi, chi)
        # Init E as 1x1 identity (1,)
        E = np.ones((1, 1), dtype=complex)
        
        for T in tensors:
            # T shape (chi_l, 2, chi_r)
            # Contract E with T and T*
            # E_new[r, r'] = sum_{l, l', s} E[l, l'] * T[l, s, r] * conj(T[l', s, r'])
            
            # Step 1: Contract E with T -> M[l', s, r] = sum_l E[l, l'] * T[l, s, r] ??
            # Wait, E indices are (l_ket, l_bra).
            # T indices (l_ket, s, r_ket).
            # T* indices (l_bra, s, r_bra).
            
            # tensordot E(l, l') with T(l, s, r) at axis 0 (l)
            # Result: (l', s, r) (indices: l_bra, phys, r_ket)
            
            # Actually E shape: (chi_ket, chi_bra)
            # T: (chi_l, 2, chi_r)
            
            # M = np.tensordot(E, T, axes=(0, 0)) # sum over chi_ket
            # M shape: (chi_bra, 2, chi_r)
            
            # Now contract with T* (chi_bra, 2, chi_r)
            # Contract M(l', s, r) with conj(T)(l', s, r') over (l', s)
            # M axes 0, 1. T* axes 0, 1.
            
            M = np.tensordot(E, T, axes=(0, 0))
            E = np.tensordot(M, T.conj(), axes=([0, 1], [0, 1])) # Result (r, r')
            
        return E[0, 0]

class QuantumCircuit:
    def __init__(self, n: int = 1, backend: str = 'vector', max_bond_dim: int = 32):
        _check_n(n)
        if backend == 'vector':
            self._backend = VectorBackend(n)
        elif backend == 'mps':
            self._backend = MPSBackend(n, max_bond_dim)
        else:
            raise ValueError(f"Unknown backend: {backend}")
            
        self.qubits_count = n
        self.classical_bits: List[int | None] = [None] * n
        
        # Legacy/Drawing support
        self.qubits: List[Qubit] = [Qubit() for _ in range(n)] 
        self.circuit: List[Tuple[str, List[int | float]]] = []
        self.__thetas: List[str] = []
        self.__measures: List[int] = []
        self.__measures_in: List[int] = []
        self.len_of_thetas = 0

    @property
    def state_vector(self):
        return self._backend.state_vector
        
    @state_vector.setter
    def state_vector(self, value):
        if hasattr(self._backend, 'state_tensor'):
             self._backend.state_vector = value
        else:
             raise NotImplementedError("Setting state vector directly not supported for this backend")

    def _apply_gate(self, gate: np.ndarray, targets: List[int]):
        self._backend.apply_gate(gate, targets)
        
    def measure(self, i: int):
        _check_index(i, self.qubits_count)
        self.circuit.append(("M", [i]))
        self.__measures_in.append(i)
        
        outcome = self._backend.measure(i)
        
        self.__measures.append(outcome)
        self.classical_bits[i] = outcome
        return outcome

    def reset(self, i: int):
        _check_index(i, self.qubits_count)
        outcome = self._backend.measure(i)
        if outcome == 1:
             self._apply_gate(X_GATE, [i])
        self.qubits[i].states = INITIAL_STATE


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
                # args[0] is qubit, args[1] is theta
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

    def reset(self, i: int):
        _check_index(i, self.qubits_count)
        # Measure and flip if 1
        prob_1 = self._get_prob_one(i)
        outcome = 1 if np.random.random() < prob_1 else 0
        self._collapse(i, outcome)
        if outcome == 1:
            self._apply_gate(X_GATE, [i])
        # Note: Original code did not add RESET to circuit list, so we don't either.
        self.qubits[i].states = INITIAL_STATE # Legacy update

    def reset_all(self):
        self.qubits = [Qubit() for _ in range(self.qubits_count)]
        self.circuit = []
        self.__thetas = []
        self.__measures = []
        self.__measures_in = []
        self.len_of_thetas = 0
        # Reset state
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

    def _get_prob_one(self, qubit):
        # Axis corresponding to qubit is (n - 1 - qubit)
        axis = self.qubits_count - 1 - qubit
        # Move axis to 0
        psi = np.moveaxis(self._state_tensor, axis, 0)
        # psi[1, ...] are amplitudes where qubit=1
        return np.sum(np.abs(psi[1])**2)

    def _collapse(self, qubit, outcome):
        axis = self.qubits_count - 1 - qubit
        # Zero out the other outcome
        other = 1 - outcome
        idx = [slice(None)] * self.qubits_count
        idx[axis] = other
        self._state_tensor[tuple(idx)] = 0
        
        # Normalize
        norm = np.linalg.norm(self._state_tensor)
        if norm > 1e-12:
            self._state_tensor /= norm

    def measure(self, i: int):
        _check_index(i, self.qubits_count)
        self.circuit.append(("M", [i]))
        self.__measures_in.append(i)
        
        prob_1 = self._get_prob_one(i)
        outcome = 1 if np.random.random() < prob_1 else 0
        
        self._collapse(i, outcome)
        self.__measures.append(outcome)
        self.classical_bits[i] = outcome
        return outcome

    def measure_all(self):
        probs = np.abs(self.state_vector) ** 2
        probs /= np.sum(probs)
        basis_state = np.random.choice(2**self.qubits_count, p=probs)
        
        # Collapse state
        new_state = np.zeros_like(self.state_vector)
        new_state[basis_state] = 1.0
        self.state_vector = new_state
        
        # Update classical bits
        # basis_state corresponds to bin string where q0 is LSB.
        # classical_bits[i] = bit at qi
        for i in range(self.qubits_count):
            self.classical_bits[i] = (basis_state >> i) & 1
            
        return bin(basis_state)[2:]

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


qc = QuantumCircuit(10).h(9).px(0).h(1)
print(qc.measure(1))
