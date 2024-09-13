# Quantipy

kill me

https://www.quantum-inspire.com/kbase/rz-gate/
https://en.wikipedia.org/wiki/List_of_quantum_logic_gates
https://en.wikipedia.org/wiki/Quantum_register
https://quantum.microsoft.com/en-us/insights/education/concepts/quantum-circuits



# Drawing ASCII circuit

```python
#store circuit attribute
circuit: List[str, List[params]] = []
```

append to circuit in every method
```python
def px(self, nqubit):
    ...
    self.circuit.append(["X", [nqubit]])
```
create a seperate list of line_str for n qubits called dg
if target = n then dg[target]
link with other entangled params as well.



