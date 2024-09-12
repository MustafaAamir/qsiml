from quantipy import QuantumCircuit
import matplotlib.pyplot as plt
from collections import Counter

def quantum_rng(bits):
    qc = QuantumCircuit(bits)
    number = 0
    for i in range(bits):
        qc.h(i)
        result = qc.measure(i)
        number |= result << i
    return number


def random_plot():
    qc = QuantumCircuit(10)
    for i in range(10):
        qc.h(i)
        print(qc.measure(i))

    # Generate a 8-bit random number
    random_number = quantum_rng(8)
    print(f"8-bit Quantum Random Number: {random_number} (binary: {random_number:08b})")

    # Generate multiple 8-bit random numbers
    num_samples = 1000
    samples = [quantum_rng(8) for _ in range(num_samples)]
    counts = Counter(samples)
    plt.bar(list(counts.keys()), list(counts.values()))
    plt.title("Distribution of Quantum Random Numbers")
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.show()

    mean = sum(samples) / num_samples
    variance = sum((x - mean) ** 2 for x in samples) / num_samples
    print(f"Mean: {mean:.2f}")
    print(f"Variance: {variance:.2f}")
