# Simulator Operations

* **Authors:** Christopher J. Wood
* **Last Updated:** 5.08.2018

---
## Table of Contents
    
1. [Operations](#operations)
2. [Standard Operations](#standard-operations)
    * [Measure](#measure)
    * [Reset](#reset)
    * [Gates](#gates)
    * [Boolean Function](#boolean-function) 
3. [Special Operations](#special-operations)
    * [Snapshots](#snapshots)
    * [Matrix multiplication](#matrix-multiplication)
    * [Noise](#noise)
        * [Readout noise](#readout-noise)
        * [Kraus noise](#kraus-noise)
        * [Unitary noise](#unitary-noise)
        * [Reset noise](#reset-noise)


---


## Operations

Circuits in Qobj are represented by a list of operations (instructions), thus it is necessary to have a having a common operation class to be passed around in qiskit-aer that has enough flexibility to encompass multiple types of operations which backends may wish to support. Most of these operations will be specified in the input QOBJ according to a JSON schema, however some operations may be for internal use only and not need to be expressed as JSON. In this document we specify the operator class used by the simulator, and give examples of specific canonical operator objects.


## Standard Operations

### Measure

Measures a subset of qubits and  record the measurement outcomes in a classical memory location. The JSON schema for measure operation is

```json
{
    "name": "measure",
    "qubits": list[int],   // qubits to measure
    "memory": list[int],   // memory bits to store outcomes in
    "register": list[int]  // (optional) register bits to store outcomes in (used for conditionals)
}
```

The `"qubits"` field specifies 1 or more qubits to measure. The `"memory"` field specifies the classical memory bit locations to store the outcomes in, it must be the same length as the qubits being measured. The optional `"register"` field is optional and can be used to specify classical register bit locations to also store the outcomes in. If present it must also be the same length of the qubits being measured. The difference between the memory and register fields is that the memory values will be returned as a `"counts"` dictionary in the results, while the register field is usually not returned, and is instead used for conditional operations.

[Back to top](#table-of-contents)


### Reset

Resets a subset of qubits to the zero state.

```json
{
    "name": "reset",
    "qubits": list[int]
}
```

The `"qubits"` field specifies 1 or more qubits to reset.

[Back to top](#table-of-contents)


### Gates

Applies the specified gate to the qubits. The JSON schema or these varies depending on the type of gate, but they all must contain a `"qubits"` field which specifies which qubits they act on, and the number of qubits must be consistent with the type of gate. 

The standard library of gates are

* Single-qubit gates with parameters: `"u0", "u1", "u2", "u3"`.
* Single-qubit gates without parameters: `"id", "x", "y", "z", "h", "s", "sdg", "t", "tdg"`
* Two-qubit gates without parameters: `"cx", "cz", "cr", "cr90"`

Single-qubit gates with parameters are specified as (for example `"u3"`)

```json
{
    "name": "u3",
    "params": [theta, phi, lam],
    "qubits": [qubit]
}
```

Gates without parameters are specified as (for example `"x"` and `"cx"`):

```json
{
    "name": "x", 
    "qubits": [qubit]
}
```

```json
{
    "name": "cx",
    "qubits": [control, target]
}
```

Note that all these gates have canonical matrix representations that must be adhered to by any simulator backend.

[Back to top](#table-of-contents)


### Boolean Function

The Boolean function (`bfunc`) operation is used for classical processing and must be supported by engines that handle conditional operations. The JSON schema specifies that boolean operations look like

```json
{
    "name": "bfunc",
    "mask": hex-string,           // hex string for bit-mask
    "relation": hex-string,       // boolean function for comparison (eg "==")
    "val": hex-string,            // hex string for val comparison
    "register": list[int], // register to store value in
    "memory": list[int]    // optional memory to store value in
}
```

where `"mask"` is a hexadecimal string of the bit-mask for the memory bits used in the function, `"relation"` is a string for the boolean comparison operator (currently only `"=="`), `"val"` is a hexadecimal string the comparison value of the function, the `"register"` field specifies 1 or more register bits to store the function outcome value in, the optional field `"memory"` is the same as the register field but stores the outcome in 1 or more memory bits.

[Back to top](#table-of-contents)


## Special Operations

### Snapshots

A *snapshot* is a special simulator instruction to records the current state of the simulator in a specified data format. The types of data that can be saved in a snapshot are:

1. The quantum state of all qubits in the system.
2. Measurement probabilities for any subset of qubits in the system.
3. Expectation values of observables (Pauli or matrix) on any subset of qubits in the system.

See [snapshots.md](snapshots.md) for details on these instructions.

[Back to top](#table-of-contents)


### Matrix multiplication

This operations allows specification of an arbitrary n-qubit matrix to be applied by the simulator. We allow two types of specification. Square matrices (`"mat"`) which are specified in JSON as:

```json
{
    "name": "mat",
    "qubits": list[int],
    "params": complex matrix, 
}
```

and diagonal matrices are specified by their diagonal

```json
{
    "name": "dmat",
    "qubits": list[int],
    "params": complex vector, 
}
```

These matrices should be unitary and when the JSON is loaded into the simulator this should be checked.

[Back to top](#table-of-contents)

### Noise

Noise is handled by inserting additional operations into a circuit to implement the noise. There are two types of noise process:
1. Noise that is *independent* of the current state of the system (*[Unitary](#unitary-noise), [Reset](#reset-noise)*)
2. Noise that is *dependent* on the current state of the system (*[Readout](#readout-noise), [Kraus](#kraus-noise)*)

The *Type 1* noises can be inserted into a sequence of operations by *sampling* a given realization of the noise from a noise model specifies the distribution for possible errors for the input operations.  This may lead to a single circuit operation sequence being modified into many different sequences for each realization of noise. Note that we do not need additional operator types to describe these noise processes --- the operators inserted to realize the noise are gates, matrix multiplication, and reset ops.

The *Type 2* noise processes cannot be sampled to choose a realization independent of the state of the system. This is because the probabilities for the classical bit-flip error for readout error, and the probabilities of choosing an individual operator to apply for Kraus error, depend on the current state classical or quantum state respectively.  Hence, we need to define new types of operators for these noise processes which can be used by the Engine/State to sample from the possible noise realizations conditional on the current simulation state.

[Back to top](#table-of-contents)


#### Kraus noise

The Kraus operation is used for noise simulations. We define it since sampling a Kraus noise operator depends on the current state value of the backend and must be handled at a lower level than the controller. The JSON for this operation is given by

```json
{
    "name": "kraus",
    "qubits": list[int],
    "params": list[complex_matrix],
}
```

where `"params"` is a list of the complex matrices for each Kraus operator.

We note that Kraus sets defined this way should be CPTP (this should be checked at generation time, not when it is applied by the backend.)

[Back to top](#table-of-contents)


#### Readout noise

The readout noise operation is an operation on classical bits (like the [boolean function](#boolean-function)). It is used to flip the recorded bit-values from a measurement where the bit-flip probability is conditional on the true mesaurement value. Since sampling the bit-flip probability depends on the measurement outcome this must be handled at run-time. The JSON for memory bits and register bits readout error are given by

```json
{
    "name": "roerror",
    "memory": list[int]
    "register": list[int]
    "params": real_matrix  // assignment fidelity matrix
}
```

Both the `"memory"` and `"register"` fields are not required, either one of both of them can be specified. The `"params"` field is a real matrix of the assignment fidelity probabilities.


Since this is a classical bit error we don't need the qubits specified in the operator, *however*, the specific matrix for assignment fidelities inserted in the operation typically will be dependent on the qubits that were measured. Also note that the assignment fidelity matrix is a probability matrix so all entries should be between [0, 1] and the columns should each sum to 1.

[Back to top](#table-of-contents)


#### Unitary noise

Unitary noise does not require any special operations. It can always be expressed as one or more [gate](#gates) operations or [matrix](#matrix-multiplication) operations.

Note, However, that because unitary noise may involve sampling the unitary to be applied from a distribution supplied by the noise model, the operation (or sequence of operations) inserted into a circuit will be for a single *realization* of the noise selected by the sampling process. 

[Back to top](#table-of-contents)


#### Reset noise

Reset noise, like unitary noise, does not require any special operations. It can be expressed as a combination of a [reset](#reset) operation specifying the state to be reset to, and optionally pre- and post-rotation [unitary](#unitary-noise) operations.

As with unitary noise, the specific choice for the state being reset to, and the choice of pre- and/or post-rotation unitary operators involves sampling from their respective distributions to choose a particular realization.

[Back to top](#table-of-contents)
