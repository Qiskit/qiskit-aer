# Snapshot Operations

* **Authors:** Christopher J. Wood
* **Last Updated:** 05.08.2018

---
## Table of Contents
    

1. [Introduction](#snapshot)
2. [Snapshot simulator instructions](#snapshot-simulator-instructions)
   1. [State snapshot](#state-snapshot)
   2. [Probabilities snapshot](#probabilities-snapshot)
   3. [Pauli observable snapshot](#pauli-observable-snapshot)
   4. [Matrix observable snapshot](#matrix-observable-snapshot)
3. [JSON schemas](#json-schemas)

---

## Introduction

A *snapshot* is a special simulator instruction to records the current state of the simulator in a specified data format. The types of data that can be saved in a snapshot are:

1. The quantum state of all qubits in the system.
2. Measurement probabilities for any subset of qubits in the system.
3. Expectation values of observables (Pauli or matrix) on any subset of qubits in the system.

The way the data is returned depends on the type of snapshot. For case 1., the return data is a list of the quantum state at the snapshot location for *each shot* of the circuit. For 2. the return data is a vector of the probabilities for measurement outcomes on the specified qubits  *averaged over all shots*, but conditional on the classical register state of the system at the snapshot location. For 3. the return data is the complex number of the observable expectation value also *averaged over all shots* conditional on the classical register value.

The result being conditional on the register value means that if there was a measurement in the circuit *before* the snapshot, which could leave a classical bit in the value of either 0 or 1, we would obtain two average quantities in the snapshot. The average over all shots where the measurement outcome was 0, and the average over all shots where the measurement outcome was 1.

**1. Quantum state snapshots:** These return a snapshot of the current state representation of the system. For example in a statevector simulator this will be a complex vector, but for a Clifford simulator this will be a Clifford tableau.


**2. Measurement probabilities snapshots:** For a snapshot on an $n$-qubit subset this will return a a dictionary containing the non-zero outcome probabilities for measurement of the the specified qubits in the *Z*-basis. Note that these probabilities are computed without actually performing the measurement, unlike the `"measure"` operation.


**3. Observables expectation value snapshots:** Records the complex expectation value for an $m$-qubit observable. For an *n*-qubit system in state $|\psi\rangle$, and an *m*-qubit observable $\mathcal{0}$, where *m* is a subset of the total *n* qubits, the expectation value of $\mathcal{O}$ is given by $\langle\mathcal{O}\rangle = \text{Tr}[\mathcal{O} \rho]$ where the $m$-qubit density matrix $\rho$ is defined by the partial trace over all qubits not in the set $m$: $\rho = \text{Tr}_{m \not\in n}[|\psi\rangle\langle\psi|]$.

The most general types of observables are when we represent the observable as a matrix: $\mathcal{O} = \sum_{i,j} m_{i,j} |i\rangle\langle j|$. From here we can consider special cases of matrices, such as:

* The matrix is a product state of subsystem matrices ($\mathcal{O} = \mathcal{O}_1\otimes\mathcal{O}_2$)
* The matrix, or subsystem matrix, is diagonal: $\mathcal{O} = \sum_{i} d_{i} |i\rangle\langle i|$
* The matrix, or subsystem matrix, is a rank-1 projector: $\mathcal{O} = |\phi\rangle\langle \phi|$
* The matrix is a Pauli matrix: $\mathcal{O} = \sum_{j} m_{j} P_j$.

All these cases have special ways of being entered in the simulator which we discuss in the following sections.

[Back to top](#table-of-contents)


## Snapshot Simulator instructions

Snapshots may be specified as a Qobj instruction with the following base JSON format:

```
{
    "name": "snapshot",
    "type": string,
    "label": string
}
```

Currently the simulator supports for snapshot types, which may be specified by the following `"type"` strings:

1. `"state"`: Snapshots of the quantum state. 
2. `"probabilities"`: Snapshots of the measurement outcome probabilities.
3. `"pauli_observable"`: Snapshots of an observable expectation value, represented as a Pauli.
3. `"matrix_observable`: Snapshots of an observable expectation value, represented as a matrix.

The `"label"` string is used to index the snapshots in the output result JSON. Only one snapshot of each type (state, probabilities, observable) may be taken for each label: If a label is repeated the later snapshot will overwrite the earlier. Pauli and Matrix snapshots are considered the same type for this purpose (an observables expectation value) and so must used different labels or they will overwrite each other. Additional fields may then be added for the parameters of the different snapshot types which we specify in this document.


### State Snapshot

The state snapshot instruction schema is given by a snapshot instruction:

```
{
    "name": "snapshot",
    "type": "state",
    "label": string
    
}
```

The format of the state in the output field will depend on the type of simulator state used. Typically it will be a complex state vector, but for specialized simulators it may take a different form (such as a Clifford table).

####  Example

Suppose we have a two-qubit system that we prepare in a maximally entangled bell state $|\psi\rangle = \frac{1}{\sqrt{2}}(|0,0\rangle + |1,1 \rangle )$ while taking snapshots between each gate used to generate the state:

```json
{
    "id": "state_snapshot_example",
    "type": "QASM",
    "experiments": [
        {   
            "config": {"shots": 1},
            "instructions": [
                {"name": "snapshot", "type": "state", "label": "initial"},
                {"name": "h", "qubits": [0]},
                {"name": "snapshot", "type": "state", "label": "middle"},
                {"name": "cx", "qubits": [0, 1]},
                {"name": "snapshot", "type": "state", "label": "final"}
            ]
        }
    ]
}
```

The result JSON will look like

```json
{
    "header": {
        "num_circuit_threads": 1,
        "num_openmp_threads": 4,
        "time_taken": 0.000467061
    },
    "id": "state_snapshot_example",
    "result": [{
        "data": {
            "snapshots": {
                "state": {
                    "final": [[[0.7071067811865476, 0.0], [0.0, 0.0], [0.0, 0.0], [0.7071067811865475, 0.0]]],
                    "initial": [[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
                    "middle": [[[0.7071067811865476, 0.0], [0.7071067811865475, 0.0], [0.0, 0.0], [0.0, 0.0]]]
                }
            }
        },
        "header": {
            "memory_sampling_opt": true,
            "seed": 1534194672,
            "shots": 1,
            "time_taken": 0.000319752
        },
        "status": "DONE",
        "success": true
    }],
    "status": "COMPLETED",
    "success": true
}
```



[Back to top](#table-of-contents)


### Probabilities snapshot

The probabilities snapshot instruction schema is given by

```
{
    "name": "snapshot",
    "type": "probabilities",
    "label": string,
    "qubits": list[int]
}
```

The list of qubits may contain any subset of qubits in the system. For example in a  2-qubit system we could have `"qubits": [0]`, `"qubits": [1]`, `"qubits": [0, 1]`. Note that the order of the qubits changes the measurement outcome as if they were measured to incrementing classical bits in the same order. The output will be a dictionary of the non-zero measurement probabilities indexed by the qubit measurement outcome. For example for the case of `"qubits": [0, 1]` the returned dictionary will be of the form: `{ "0x0": P(q1=0, q0=0), "0x1": P(q1=0, q0=1), "0x2": P(q1=1, q0=0), "0x3": P(q1=1, q0=1) }`.

If the circuit contains classical registers and measurements before the snapshot, the returned dictionary will be shown conditional on the memory classical register state.


####  Example

Suppose we have a two-qubit system in the maximally entangled bell state $|\psi\rangle = \frac{1}{\sqrt{2}}(|0,0\rangle + |1,1 \rangle )$ and apply a measurement on qubit-0. Suppose we take a probabilities snapshot before, and after the measurement. The input Qobj JSON is given by:

```json
{
    "id": "probabilities_snapshot_example",
    "type": "QASM",
    "experiments": [
        {   
            "config": {"shots": 1},
            "instructions": [
                {"name": "h", "qubits": [0]},
                {"name": "cx", "qubits": [0, 1]},
                {"name": "snapshot", "type": "probabilities",
                    "label": "pre_measure", "qubits": [1, 0]},
                {"name": "measure", "qubits": [0, 1], "memory": [0, 1]},
                {"name": "snapshot", "type": "probabilities",
                    "label": "post_measure", "qubits": [1, 0]}
            ]
        }
    ]
}
```

The result JSON will look like


```json
{
    "header": {
        "num_circuit_threads": 1,
        "num_openmp_threads": 4,
        "time_taken": 0.008081267000000001
    },
    "id": "probabilities_snapshot_example",
    "result": [{
        "data": {
            "counts": {
                "0x0": 508,
                "0x3": 492
            },
            "snapshots": {
                "probabilities": {
                    "post_measure": [{
                        "memory": "0x0",
                        "values": {
                            "0x0": 1.0
                        }
                    }, {
                        "memory": "0x3",
                        "values": {
                            "0x3": 1.0
                        }
                    }],
                    "pre_measure": [{
                        "memory": "0x0",
                        "values": {
                            "0x0": 0.5000000000000001,
                            "0x3": 0.49999999999999994
                        }
                    }]
                }
            }
        },
        "header": {
            "seed": 3399576275,
            "shots": 1000,
            "time_taken": 0.007999758000000001
        },
        "status": "DONE",
        "success": true
    }],
    "status": "COMPLETED",
    "success": true
}
```

[Back to top](#table-of-contents)


### Pauli observable snapshot

Pauli observables are a special case of general matrix observables where the matrix may be written in terms of the Pauli-operator basis: $\mathcal{O}_k = \bigotimes_{j=1}^n P_j$ where $P_j \in \{I, X, Y, Z\}$.  The JSON instruction is given by

```
{
    "name": "snapshot",
    "type": "pauli_observable",
    "label": string,
    "params": [
        {"coeff": complex, "qubits": list[int], "op": pauli_string},
        ...
    ]
}
```

Where for a Pauli observable $\mathcal{O}_n = \bigotimes_{j=1}^k w_j P_j$ the params are a list of components each given as `{"coeff": [real(w_j), imag(w_j)]`, `"op": P_j}`. For an $n$-qubit Pauli the `P_j` operations are represented as a string containing *n* characters from `{"I", "X", "Y", "Z"}`, where the string order corresponds to the list order. Eg for 3-qubits with `"qubits": [2, 0, 1]` the string `XYZ` corresponds to the operator $X_2 \otimes Z_1 \otimes Y_0$. If the circuit contains classical registers and measurements before the snapshot, the returned dictionary will be shown conditional on the memory classical register state.

**Implementation details:** In the simulator the computation of a Pauli expectation value is implemented by making a copy of the current system state, then applying each Pauli operator as a single-qubit gate and then computing the inner product of this state with the original state. This is done for each component and they are accumulated with their respective coefficients. Simple caching is done so that if multiple Pauli snapshots are placed after each other, each Pauli operator component expectation value $\langle P_j \rangle$ will only be computed once.


####  Example


Let us consider the sample example from the [probabilities snapshot](#probabilities-snapshot), with the snapshots of the two Pauli observables `ZZ` and `ZI+IZ` before and after the measurement. The input Qobj JSON is given by:

```json
{
    "id": "pauli_observable_snapshot_example",
    "type": "QASM",
    "experiments": [
        {   
            "config": {"shots": 1},
            "instructions": [
                {"name": "h", "qubits": [0]},
                {"name": "cx", "qubits": [0, 1]},
                {"name": "snapshot", "type": "pauli_observable",
                 "label": "<ZZ>pre_measure",
                 "params": [{"coeff": 1, "qubits": [1, 0], "op": "ZZ"}]},
                {"name": "snapshot", "type": "pauli_observable",
                 "label": "<ZI+IZ>pre_measure",
                 "params": [{"coeff": 1, "qubits": [1, 0], "op": "ZI"},
                            {"coeff": 1, "qubits": [1, 0], "op": "IZ"}]},
                {"name": "measure", "qubits": [0, 1], "memory": [0, 1]},
                {"name": "snapshot", "type": "pauli_observable",
                    "label": "<ZZ>post_measure",
                    "params": [{"coeff": 1, "qubits": [1, 0], "op": "ZZ"}]},
                   {"name": "snapshot", "type": "pauli_observable",
                    "label": "<ZI+IZ>post_measure",
                    "params": [{"coeff": 1, "qubits": [1, 0], "op": "ZI"},
                               {"coeff": 1, "qubits": [1, 0], "op": "IZ"}]}
            ]
        }
    ]
}
```

The result JSON will look like

```json
{
    "header": {
        "num_circuit_threads": 1,
        "num_openmp_threads": 4,
        "time_taken": 0.016819449
    },
    "id": "pauli_observable_snapshot_example",
    "result": [{
        "data": {
            "counts": {
                "0x0": 494,
                "0x3": 506
            },
            "snapshots": {
                "observables": {
                    "<ZI+IZ>post_measure": [{
                        "memory": "0x0",
                        "value": [2.0, 0.0]
                    }, {
                        "memory": "0x3",
                        "value": [-1.9999999999999998, 0.0]
                    }],
                    "<ZI+IZ>pre_measure": [{
                        "memory": "0x0",
                        "value": [3.993469235085526e-16, 0.0]
                    }],
                    "<ZZ>post_measure": [{
                        "memory": "0x0",
                        "value": [1.0, 0.0]
                    }, {
                        "memory": "0x3",
                        "value": [0.9999999999999999, 0.0]
                    }],
                    "<ZZ>pre_measure": [{
                        "memory": "0x0",
                        "value": [1.0, 0.0]
                    }]
                }
            }
        },
        "header": {
            "seed": 1798320531,
            "shots": 1000,
            "time_taken": 0.016716694
        },
        "status": "DONE",
        "success": true
    }],
    "status": "COMPLETED",
    "success": true
}
```



[Back to top](#table-of-contents)



### Matrix observable snapshot

If Pauli operators are insufficient, general matrix observables can be defined for subsets of qubits in the system. Matrices my be represented by subsystem components in a tensor product structure: Eg for a matrix $M = A\otimes B$, the matrices $A$, and $B$ may be passed in separately, along with which qubits the sub-matrices act on. The general JSON schema for this operation is:

```
{
    "name": "snapshot",
    "type": "matrix_observable",
    "label": string,
    "params": [
        {"coeff": complex, "qubits": list[list[int]], "op": list[complex matrix]},
         ...
    ]
}
```

The `"qubits"` specifies a list of the component subsystems, and the `"ops"` field the matrices for each of these subsystems. For example: If we have a 3-qubit matrix $M = A_{2,1} \otimes B_0$ where subscripts denote the system these act on, we may enter this as: `{"coeff": [1, 0], "qubits": [[2,1],[0]], "op": [A, B]}`. For a single matrix the lists will be of length 1.

For the sub-matrix components, they may be specified as either:

1. *N x N* complex-matrix
2. *1 x N* row-vector matrix that stores the diagonal of a diagonal matrix.
3. *N x 1* column-vector matrix that stores the vector $|v\rangle$ of a rank-1 projector $|v\rangle\langle v |$.


**Implementation details:** In the simulator the computation of a Matrix expectation value is implemented by making a copy of the current system state, then applying each sub-matrix operator as an m-qubit gate on the sub-qubits for that component, and then computing the inner product of this state with the original state. This is done for each component and they are accumulated with their respective coefficients.

####  Example

Let us consider the sample example from the [Pauli observable snapshot](#pauli-observable-snapshot), with the snapshots of the Pauli observables `ZZ` and `ZI+IZ` before and after the measurement. The input Qobj JSON is given by:

```json
{
    "id": "matrix_observable_snapshot_example",
    "type": "QASM",
    "experiments": [
        {   
            "config": {"shots": 1},
            "instructions": [
                {"name": "h", "qubits": [0]},
                {"name": "cx", "qubits": [0, 1]},
                {"name": "snapshot", "type": "matrix_observable",
                 "label": "<ZZ>pre_measure",
                 "params": [{"coeff": 1, "qubits": [[1], [0]], "ops": [
                                [[[1, 0], [0, 0]], [[0, 0], [0, -1]]],
                                [[[1, 0], [0, 0]], [[1, 0], [0, -1]]]
                            ]}]},
                {"name": "snapshot", "type": "matrix_observable",
                 "label": "<ZI+IZ>pre_measure",
                 "params": [{"coeff": 1, "qubits": [[1]], "ops": [
                            [[[1, 0], [0, 0]], [[0, 0], [0, -1]]]]},
                            {"coeff": 1, "qubits": [[0]], "ops": [
                                [[[1, 0], [0, 0]], [[0, 0], [0, -1]]]]}
                            ]},
                {"name": "measure", "qubits": [0, 1], "memory": [0, 1]},
                {"name": "snapshot", "type": "matrix_observable",
                    "label": "<ZZ>post_measure",
                    "params": [{"coeff": 1, "qubits": [[1], [0]], "ops": [
                        [[[1, 0], [0, 0]], [[0, 0], [0, -1]]],
                        [[[1, 0], [0, 0]], [[1, 0], [0, -1]]]
                    ]}]},
                   {"name": "snapshot", "type": "matrix_observable",
                    "label": "<ZI+IZ>post_measure",
                    "params": [{"coeff": 1, "qubits": [[1]], "ops": [
                        [[[1, 0], [0, 0]], [[0, 0], [0, -1]]]]},
                        {"coeff": 1, "qubits": [[0]], "ops": [
                            [[[1, 0], [0, 0]], [[0, 0], [0, -1]]]]}
                        ]}
            ]
        }
    ]
}
```

The result JSON will look like

```json
{
    "header": {
        "num_circuit_threads": 1,
        "num_openmp_threads": 4,
        "time_taken": 0.013922274
    },
    "id": "matrix_observable_snapshot_example",
    "result": [{
        "data": {
            "counts": {
                "0x0": 506,
                "0x3": 494
            },
            "snapshots": {
                "observables": {
                    "<ZI+IZ>post_measure": [{
                        "memory": "0x0",
                        "value": [1.9999999999999998, 0.0]
                    }, {
                        "memory": "0x3",
                        "value": [0.0, -2.0]
                    }],
                    "<ZI+IZ>pre_measure": [{
                        "memory": "0x0",
                        "value": [1.0000000000000002, -0.9999999999999999]
                    }],
                    "<ZZ>post_measure": [{
                        "memory": "0x0",
                        "value": [0.9999999999999999, 0.0]
                    }, {
                        "memory": "0x3",
                        "value": [-1.0, 0.0]
                    }],
                    "<ZZ>pre_measure": [{
                        "memory": "0x0",
                        "value": [1.996734617542763e-16, 0.0]
                    }]
                }
            }
        },
        "header": {
            "seed": 1670553245,
            "shots": 1000,
            "time_taken": 0.013792974000000001
        },
        "status": "DONE",
        "success": true
    }],
    "status": "COMPLETED",
    "success": true
}
```

[Back to top](#table-of-contents)

## JSON Schemas

#### Snapshot base definition

```json
{"snapshot": {
    "type": "object",
    "properties": {
    "name": {
        "type": "string",
        "enum": ["snapshot"]
    },
    "type": {
        "type": "string", 
        "enum": ["state", "probabilities", "pauli_observable", "matrix_observable"]
    },
    "label": {"type": "string"}
    },
    "required": ["name", "type", "label"]
}}
```

#### State snapshot definition

```json
{"snapshot_state": {
    "allOf": [
        {"$ref": "#/snapshot"},
        {"properties": {"type": {"type": "string", "enum": ["state"]}}}
]}}
```

#### Probabilities snapshot definition

```json
{"snapshot_probs": {
    "allOf": [
        {"$ref": "#/snapshot"},
        {"properties": {
            "type": {"type": "string", "enum": ["probabilities"]}},
            "qubits": {
                "type": "array",
                "uniqueItems": true,
                "minItems": 1,
                "items": {"type": "integer"}
}}]}}
```

#### Pauli observable snapshot definition

```json
{"snapshot_pauli": {
    "allOf": [
        {"$ref": "#/snapshot"},
        {"properties": {
            "type": {"type": "string", "enum": ["pauli_observable"]}},
            "params": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object", 
                    "properties": {
                        "coeffs": {
                            "oneOf": [{"type": "number"}, {"$ref": "#/complex"}]
                        },
                        "qubits": {
                            "type": "array",
                            "uniqueItems": true,
                            "minItems": 1,
                            "items": {"type": "integer"}
                        },
                        "op": {"type": "string", "pattern": "[IXYZ]+"}
}}}}]}}
```

#### Matrix observable snapshot definition

```json
{"snapshot_pauli": {
    "allOf": [
        {"$ref": "#/snapshot"},
        {"properties": {
            "type": {"type": "string", "enum": ["matrix_observable"]}},
            "params": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object", 
                    "properties": {
                        "coeffs": {
                            "oneOf": [{"type": "number"}, {"$ref": "#/complex"}]
                        },
                        "qubits": {
                            "type": "array",
                            "uniqueItems": true,
                            "minItems": 1,
                            "items": {"type": "array", 
                                      "uniqueItems": true,
                                      "minItems": 1,
                                      "items": "integer"}
                        },
                        "ops": {"type": "array",
                                "minItems": 1,
                                "items": {"$ref": "#/complex_matrix"}}
}}}}]}}
```