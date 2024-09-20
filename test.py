import unittest
from quasi import QuantumCircuit
import cmath

HALF_SQRT = complex((1 / 2) ** 0.5)
COMPLEX_ZERO = complex(0)
COMPLEX_ONE = complex(1)
INITIAL_STATE = [COMPLEX_ONE, COMPLEX_ZERO]
ZERO_STATE = INITIAL_STATE
ONE_STATE = [COMPLEX_ZERO, COMPLEX_ONE]
ONE_SQRT2 = [complex(1 / cmath.sqrt(2)), complex(1 / cmath.sqrt(2))]


class TestGates(unittest.TestCase):
    """
    testing hadamard:
    applying it once results quibits being HALF_SQRT
    aPplying it twice results in quibits reverting to their original state
    """

    def test_hadamard(self):
        qc = QuantumCircuit(1)
        # storing states for later comparison
        zeroeth = qc.qubits[0].states[0]
        oneth = qc.qubits[0].states[1]
        qc.h(0)
        self.assertAlmostEqual(HALF_SQRT, qc.qubits[0].states[0])
        self.assertAlmostEqual(HALF_SQRT, qc.qubits[0].states[1])
        qc.h(0)
        self.assertAlmostEqual(zeroeth, qc.qubits[0].states[0])
        self.assertAlmostEqual(oneth, qc.qubits[0].states[1])

    def test_px(self):
        qc = QuantumCircuit(1)

        qc.px(0)
        self.assertAlmostEqual(0, qc.qubits[0].states[0])
        self.assertAlmostEqual(1, qc.qubits[0].states[1])
        qc.px(0)
        self.assertAlmostEqual(1, qc.qubits[0].states[0])
        self.assertAlmostEqual(0, qc.qubits[0].states[1])

    def test_py(self):
        qc = QuantumCircuit(2)
        qc.px(1)
        qc.py(0)
        self.assertAlmostEqual(0, qc.qubits[0].states[0])
        self.assertAlmostEqual(1j, qc.qubits[0].states[1])
        qc.py(1)
        self.assertAlmostEqual(-1j, qc.qubits[1].states[0])
        self.assertAlmostEqual(0, qc.qubits[1].states[1])

    def test_pz(self):
        qc = QuantumCircuit(2)
        qc.px(1)
        qc.pz(0)
        self.assertAlmostEqual(1, qc.qubits[0].states[0])
        self.assertAlmostEqual(0, qc.qubits[0].states[1])
        qc.pz(1)
        self.assertAlmostEqual(0, qc.qubits[1].states[0])
        self.assertAlmostEqual(-1, qc.qubits[1].states[1])

    def test_phase_no_px(self):
        qc = QuantumCircuit(5)
        pi_values = [0.0, cmath.pi / 2, cmath.pi, (1.5) * cmath.pi, 2 * cmath.pi]
        for i in range(5):
            qc.phase(i, pi_values[i])
            self.assertAlmostEqual(qc.qubits[i].states[1], COMPLEX_ZERO)

    def test_phase_px(self):
        qc = QuantumCircuit(5)
        pi_values = [0.0, cmath.pi / 2, cmath.pi, (1.5) * cmath.pi, 2 * cmath.pi]
        # i added some changes to phase
        # if theta % pi then use the real part only (-1 or 1)
        # else if theta % cmath.pi/2 then use the imaginary part only (-1j or 1j)

        euler_values = [
            COMPLEX_ONE,
            complex(0 + 1j),
            -COMPLEX_ONE,
            complex(0 - 1j),
            COMPLEX_ONE,
        ]
        for i in range(5):
            qc.px(i)
            qc.phase(i, pi_values[i])
            self.assertAlmostEqual(qc.qubits[i].states[1].real, euler_values[i].real)

        qc = QuantumCircuit(360)
        for i in range(360, 0, -1):
            qc.px(i - 1)
            qc.phase(i - 1, (2 * (cmath.pi / i)))
            self.assertAlmostEqual(
                qc.qubits[i - 1].states[1], cmath.exp(1j * (2 * (cmath.pi / i)))
            )

    def test_measure(self):
        qc = QuantumCircuit(100)
        for i in range(100):
            self.assertEqual(qc.measure(i), 0)

        qc = QuantumCircuit(100)
        for i in range(100):
            qc.px(i)
            self.assertEqual(qc.measure(i), 1)

    def test_cnot(self):
        """flips target qubit only if control qubitis one"""
        qc = QuantumCircuit(2)
        qc.cnot(0, 1)
        self.assertEqual(qc.qubits[1], INITIAL_STATE)
        qc.px(0)
        qc.cnot(0, 1)
        self.assertEqual(qc.qubits[1], ONE_STATE)
        qc.cnot(1, 0)
        self.assertEqual(qc.qubits[0], ZERO_STATE)

    def test_ccnot(self):
        """if both control bits are one, flips target"""
        qc = QuantumCircuit(3)
        qc.ccnot(0, 1, 2)
        self.assertEqual(qc.qubits[2], INITIAL_STATE)

        qc.px(0)
        qc.ccnot(0, 1, 2)
        self.assertEqual(qc.qubits[2], INITIAL_STATE)

        qc.px(1)
        qc.ccnot(0, 1, 2)
        self.assertEqual(qc.qubits[2], ONE_STATE)
        """finally changes when both control bits are one"""

        qc.ccnot(1, 2, 0)
        self.assertEqual(qc.qubits[0], INITIAL_STATE)

    def test_cswap(self):
        """if control bit is one, swaps two target bits"""
        qc = QuantumCircuit(3)
        qc.px(2)
        qc.cswap(0, 1, 2)
        self.assertEqual(qc.qubits[1], INITIAL_STATE)
        self.assertEqual(qc.qubits[2], ONE_STATE)

        qc.px(0)
        qc.cswap(0, 1, 2)
        self.assertEqual(qc.qubits[1], ONE_STATE)
        self.assertEqual(qc.qubits[2], ZERO_STATE)

    def test_rx(self):
        qc = QuantumCircuit(1)
        for i in range(360, 0, -1):
            theta = 2 * (cmath.pi / i)
            qc.qubits[0] = INITIAL_STATE
            qc.rx(0, theta)
            self.assertAlmostEqual(qc.qubits[0].states[0], cmath.cos(theta / 2))
            self.assertAlmostEqual(qc.qubits[0].states[1], -1j * cmath.sin(theta / 2))
            qc.qubits[0] = ONE_STATE
            qc.rx(0, theta)
            self.assertAlmostEqual(qc.qubits[0].states[0], -1j * cmath.sin(theta / 2))
            self.assertAlmostEqual(qc.qubits[0].states[1], cmath.cos(theta / 2))

    def test_ry(self):
        qc = QuantumCircuit(1)
        for i in range(360, 0, -1):
            theta = 2 * (cmath.pi / i)
            qc.qubits[0] = INITIAL_STATE
            qc.ry(0, theta)
            self.assertAlmostEqual(qc.qubits[0].states[0], cmath.cos(theta / 2))
            self.assertAlmostEqual(qc.qubits[0].states[1], cmath.sin(theta / 2))
            qc.qubits[0] = ONE_STATE
            qc.ry(0, theta)
            self.assertAlmostEqual(qc.qubits[0].states[0], -cmath.sin(theta / 2))
            self.assertAlmostEqual(qc.qubits[0].states[1], cmath.cos(theta / 2))

    def test_rz(self):
        qc = QuantumCircuit(1)
        for i in range(360, 0, -1):
            theta = 2 * (cmath.pi / i)
            qc.qubits[0] = INITIAL_STATE
            qc.rz(0, theta)
            self.assertAlmostEqual(
                qc.qubits[0].states[0], cmath.cos(theta / 2) + cmath.sin(theta / 2) * -1j
            )
            self.assertAlmostEqual(qc.qubits[0].states[1], 0)
            qc.qubits[0] = ONE_STATE
            qc.rz(0, theta)
            self.assertAlmostEqual(qc.qubits[0].states[0], 0)
            self.assertAlmostEqual(
                qc.qubits[0].states[1], cmath.cos(theta / 2) + cmath.sin(theta / 2) * 1j
            )

    """
    The rotation matrices are related to the Pauli matrices in the following way:
    rx(π) = −iX , ry(π) = −iY , rz(π) = −iZ
    """

    def test_rz_equal_pauliz(self):
        """-1j * pauliZ() should equal rz(pi)"""
        qc = QuantumCircuit(2)
        qc.rz(0, cmath.pi)
        qc.pz(1)
        self.assertAlmostEqual(qc.qubits[0].states[0], -1j * qc.qubits[1].states[0])
        self.assertAlmostEqual(qc.qubits[0].states[1], -1j * qc.qubits[1].states[1])

    def test_rx_equal_paulix(self):
        """rx(pi) should equal paulix"""
        qc = QuantumCircuit(2)
        qc.rx(0, cmath.pi)
        qc.px(1)
        self.assertAlmostEqual(qc.qubits[0].states[0], -1j * qc.qubits[1].states[0])
        self.assertAlmostEqual(qc.qubits[0].states[1], -1j * qc.qubits[1].states[1])

    def test_ry_equal_pauliy(self):
        """ry(pi) should equal -i * paulix"""
        qc = QuantumCircuit(2)
        qc.ry(0, cmath.pi)
        qc.py(1)
        self.assertAlmostEqual(qc.qubits[0].states[0], -1j * qc.qubits[1].states[0])
        self.assertAlmostEqual(qc.qubits[0].states[1], -1j * qc.qubits[1].states[1])

    def rz_phase_equal(self):
        qc = QuantumCircuit(2)
        theta = 12
        qc.rz(0, theta)
        qc.phase(0, theta / 2)
        qc.phase(1, theta)
        self.assertAlmostEqual(qc.qubits[0].states[0], qc.qubits[1].states[0])
        self.assertAlmostEqual(qc.qubits[0].states[1], qc.qubits[1].states[1])

    """
    Testing other identities mentioned in:
    https://en.wikipedia.org/wiki/List_of_quantum_logic_gates
    """
    def test_dump(self):
        '''
        compare to Q# DrawMachine function
        '''
        pass


class TestDraw(unittest.TestCase):
    def test_aggregate(self):
        """
        |q0⟩————Y—————⨁———————
        |q1⟩—X————————●——●——●—
        |q2⟩———————Z—————⨁——│—
        |q3⟩————————————————⨁—
        """
        qc = QuantumCircuit(4)
        qc.px(1)
        qc.py(0)
        qc.pz(2)
        qc.cnot(1, 0)
        qc.cnot(1, 2)
        qc.ccnot(0, 1, 3)
        qc.draw("test_aggregate")

        """
        using ╂ to denote an entangled qubit in the middle.
        Possible problem is that entagled bits can't be connected
        via the | when the distance is more than 1 qubit, tho it's easy
        to infer.
        |q0⟩————Y—————⨁—————●—
        |q1⟩—X————————●——●——╂—
        |q2⟩———————Z—————⨁————
        |q3⟩————————————————⨁—
        Possible solution could be storing each qubit line as a string,
        if the current qubit is not in any targets AND it's between
        two entangled bits then add "-|-" otherwise "---"
"""
class TestStates(unittest.TestCase):
    """
    hceck if i'm calculating the state vector correctly
    """
    def test_first(self):
        """
        should be
        1000, 1101
        not
        1000, 1100
        """
        qc = QuantumCircuit(4)
        qc.px(1)
        qc.px(0)
        qc.px(2)
        qc.cnot(0, 2)
        qc.h(1)
        qc.ccnot(0, 1, 3)
        qc.operations("Testing states")
        print()
        print(qc)
        qc.dump()

        """
        first px to qubit 1
        [0j, 0j, 0j, 0j, (1+0j), 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j]
        (correct)
        ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
        ┃ Basis states   ┃ |100⟩      ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━┫
        ┃ Amplitudes     ┃ 1.0 + 0.0i ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━┫
        ┃ Probabilties   ┃ 100.00%    ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━┫
        ┃ Phases         ┃ 0π         ┃
        ┗━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
        px to qubit 0
        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, (1+0j), 0j, 0j, 0j]
        (correct)
        ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
        ┃ Basis states   ┃ |1100⟩     ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━┫
        ┃ Amplitudes     ┃ 1.0 + 0.0i ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━┫
        ┃ Probabilties   ┃ 100.00%    ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━┫
        ┃ Phases         ┃ 0π         ┃
        ┗━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
        px to qubit 2
        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, (1+0j), 0j]
        (correct)
        ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
        ┃ Basis states   ┃ |1110⟩     ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━┫
        ┃ Amplitudes     ┃ 1.0 + 0.0i ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━┫
        ┃ Probabilties   ┃ 100.00%    ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━┫
        ┃ Phases         ┃ 0π         ┃
        ┗━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
        ccnot (0, 2)
        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, (1+0j), 0j, 0j, 0j]
        (correct)
        ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
        ┃ Basis states   ┃ |1100⟩     ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━┫
        ┃ Amplitudes     ┃ 1.0 + 0.0i ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━┫
        ┃ Probabilties   ┃ 100.00%    ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━┫
        ┃ Phases         ┃ 0π         ┃
        ┗━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━┛
        Hadamard to qubit 1
        [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, (0.7071067811865476+0j), 0j, 0j, 0j, (-0.7071067811865476+0j), 0j, 0j, 0j]
        (correct)
        ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
        ┃ Basis states   ┃ |1000⟩        ┃ |1100⟩         ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━┫
        ┃ Amplitudes     ┃ 0.7071 + 0.0i ┃ -0.7071 + 0.0i ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━┫
        ┃ Probabilties   ┃ 50.00%        ┃ 50.00%         ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━┫
        ┃ Phases         ┃ 0π            ┃ 1π             ┃
        ┗━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━┛
        ccnot 0 1 3
        [0, 0, 0, 0, 0, 0, 0, 0, (0.7071067811865476+0), 0, 0, 0, (-0.7071067811865476+0), 0, 0, 0]
        (INCORRECT)

        ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
        ┃ Basis states   ┃ |1000⟩        ┃ |1100⟩         ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━┫
        ┃ Amplitudes     ┃ 0.7071 + 0.0i ┃ -0.7071 + 0.0i ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━┫
        ┃ Probabilties   ┃ 50.00%        ┃ 50.00%         ┃
        ┣━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━┫
        ┃ Phases         ┃ 0π            ┃ 1π             ┃
        ┗━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━┛

        solving for the kronecker product
          [0, 1] ⊗ [1/(2)^0.5, -1/(2)^0.5] = [0, 0, 1/sqrt2, -1/sqrt2]
          [1, 0] ⊗ [1/(2)^0.5, -1/(2)^0.5] = [0, 0, 0, 0, 1/sqrt2, 0, -1/sqrt2, 0]
          [1, 0] ⊗ [0, 0, 0, 0, 1/sqrt2, 0, -1/sqrt2, 0] =
          [0, 0, 0, 0, 0, 0, 0, 0, 1/sqrt2, 0, 0, 0, -1/sqrt2, 0, 0, 0]

          comparing it with the one we got
          [0, 0, 0, 0, 0, 0, 0, 0, 1/sqrt2, 0, 0, 0, -1/sqrt2, 0, 0, 0]

        therefore i'm getting the correct tensor product
        has to be an index error or something

        UNLESS

        We entangled qubits 0 and 2 by applying a cnot beforehand
        0, while entangled with 2, is used as a control bit for 3 as well

        for reasons beyond my understanding, the state of qubit[3] should not flip
        instead of this (q[3] ⊗ state_vector)
          [1, 0] ⊗ [0, 0, 0, 0, 1/sqrt2, 0, -1/sqrt2, 0] =
        it should be
          [0, 1] ⊗ [0, 0, 0, 0, 1/sqrt2, 0, -1/sqrt2, 0] =

        ??????

        this gives us
        [0,0,0,0,0,0,0,0,0,1/sqrt2, 0, 0, 0, -1/sqrt2, 0, 0]

        okay never mind, the second state is in the right place (1101)
        but the first one is ahead by one

        kmsing
        """
