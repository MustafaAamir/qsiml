def print_circuit(self):
        """Print an ASCII representation of the quantum circuit.
        circuit_operations=[('Gate',[target]),('Gate',[target]),('Gate',[target])]"""

        circuit=self.circuit_operations
        gate_symbols = {
            'H': 'H',
            'X': 'X',
            'Y': 'Y',
            'M': 'M',
            'Z': 'Z',
            'CNOT': 'X',
            'SWAP': 'x',
            'CSWAP': 'x',
            'Rx': 'Rx',
            'Ry': 'Ry',
            'Rz': 'Rz',
            'CCNOT': 'X',
            # Add more gate symbols as needed
        }

        num_qubits = len(self.qubits)
        num_gates = len(circuit)

        # Print the header


        # Print the gates and qubits
        for qubit in range(num_qubits):
            print(f"q{qubit}|", end="")
            for gate_index in range(num_gates):
                if qubit in circuit[gate_index][1]:
                    if len(circuit[gate_index][1]) > 1 and qubit == circuit[gate_index][1][0]:
                        print(f"-.-", end="")
                    else:
                        if circuit[gate_index][0] in ('Rx','Ry','Rz'):
                            print(f"-{gate_symbols[circuit[gate_index][0]]}", end="")
                        else:
                            print(f"-{gate_symbols[circuit[gate_index][0]]}-", end="") 
                else:
                    print("---", end="")
            print()