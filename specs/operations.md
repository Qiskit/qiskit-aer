# Operation Class Specification

* **Authors:** Christopher J. Wood
* **Last Updated:** 27.07.2018

---
## Table of Contents
    
1. [Operation Class](#operation-class)
2. [QASM Operations](#standard-operations)
    * [Measure](#measure)
    * [Reset](#reset)
    * [Gates](#gates)
3. [General Operations](#special-operations)
    * [Boolean Function](#boolean-function)  
    * [Matrix Multiplication](#matrix-multiplication)
    * [Snapshot](#snapshot)
4. [Observables Operations](#observables-operations)
   * [Matrix Observables](#matrix-observables)
   * [Pauli Matrix Observables](#pauli-matrix-observables)
   * [Measurement Observables](#measurement-observables)
5. [Noise Operations](#noise-operations)
    * [Readout Noise](#readout-noise)
    * [Kraus Noise](#kraus-noise)
    * [Unitary Noise](#unitary-noise)
    * [Reset Noise](#reset-noise)

---
## Operation Class

Circuits are represented by a vector of operations, thus it is necessary to have a having a common operation class to be passed around in qiskit-aer that has enough flexibility to encompass multiple types of operations which backends may wish to support. Most of these operations will be specified in the input QOBJ according to a JSON schema, however some operations may be for internal use only and not need to be expressed as JSON. In this document we specify the operator class used by the simulator, and give examples of specific canonical operator objects.



### JSON Schema

**Open Question:** *Do we store square matrices as a nested list of list of complex values, or as a column-major vectorized list of complex values?*

Operations in a QOBJ must conform to the following JSON schema (with a couple of exceptions such as snapshot).


```
{
    "name": str,
    "qubits": list[int],
    "params": list[varies],
    "textparams": list[str],
    "conditional": int
}
```

### C++ Class

Representing this as a C++ data structure we define a general operator class `Op`, which can take params in multiple different formats (which may needed internally for different operations)

```cpp
using namespace std;
using int_t = int64_t;
using uint_t = uint64_t;
using reg_t = vector<uint_t>;
using complex_t = complex<double>;
using cvector_t = vector<complex_t>;
using cmatrix_t = matrix<complex_t>;

struct Op {

    string name;           // operation name
    reg_t qubits;          // (opt) qubits operation acts on
    reg_t registers;       // (opt) register locations it acts on (measurement, conditional)
    reg_t memory;          // (opt) register operation it acts on (measurement)
    uint_t conditional;    // (opt) the (single) register location to look up for conditional
    
    // Data structures for parameters

    vector<string> params_s;     // string params
    vector<double> params_d;     // double params
    vector<complex_t> params_z;  // complex double params
    vector<cvector_t> params_v;  // complex vector params
    vector<cmatrix_t> params_m;  // complex matrix params
}
```

[Back to top](#table-of-contents)

---
## Standard Operations

### Measure

#### JSON Schema

```
{
    "name": "measure",
    "qubits": list[int]   // qubits to measure
    "memory": list[int]   // (optional) classical bits to store result in
    "register": list[int] // (optional) for conditionals
}
```

For the simulator a measurement must specify qubits to measure, and optionally bit locations in a classical memory, and/or classical register to store the measurement outcomes. Both these fields are optional and if they are empty the measurement outcome is ignored. If they exist they must be the number of bits in the field must be the same length as the number of qubits measured.

#### C++ Object

A `measure` Op is specified as follows

```cpp
Op meas;
meas.name = "measure";                     // measure identitier
meas.qubits = {qubits, ... };              // qubits to measure
meas.memory = {memory bits, ... };         // (optional) memory bits to store result in
meas.registers = {registers bits, ... };   // (optional) register bits to store output in

```

[Back to top](#table-of-contents)


### Reset

#### JSON Schema

```
{
    "name": "reset",
    "qubits": list[int]  // qubits to reset
}
```

#### C++ Object

Internally reset operations are more general than their QOBJ json counterparts. They allow the specification of the state to be Z-basis eigenstate to be reset to. Note that this field is not optional and when parsing the JSON qobj must be initialized to all zeros.

A `reset` Op is specified as follows:

```cpp
Op reset;
reset.name = "reset";             // reset identitier
reset.qubits = {qubits, ... };    // qubits to measure
reset.params_d = {states, ...};   // memory bits to store result in
```

For additional examples. For a 2-qubit state the reset operation can specify the final state as `reset.params_d = [i_0, i_1]` for the $|i_1, i_0\rangle$ state.

[Back to top](#table-of-contents)


### Gates

**Open Question:** *Should we forbid gates `"U"` and `"CX"` and only support `"u3"` and `"cx"` versions instead? Currently I am leaning towards yes, support only the minimal ones.*

**TODO:** *List canonical matrix representation for all named gates.*

Gates must specify which qubits they act on, and the nubmer of qubits must be consistant with the type of gate. 

* Single-qubit gates with parameters: `"u0", "u1", "u2", "u3"`.
* Single-qubit gates without parameters: `"id", "x", "y", "z", "h", "s", "sdg", "t", "tdg"`
* Two-qubit gates without parameters: `"cx", "cz", "cr", "cr90"`

Note that all these gates have canonincal matrix representations that must be adhered to by any simulator backend.

For multiplication by custom matrices also see [matrix multiplication](#matrix-multiplication) operations.

#### JSON Schema

Single-qubit gates with parameters are specified as (for example)

```
{
    "name": "u3",                // u3 gate
    "params": [theta, phi, lam], // double parameters
    "qubits": [qubit]            // single-qubit to apply gate to
}
```

Gates without parameters are specified as (for example):

```
{
    "name": "x",       // Pauli-X gate
    "qubits": [qubit]  // qubit to apply gate to
}
```

```
{
    "name": "cx",                // cx (CNOT) gate
    "qubits": [control, target]  // control and target qubits for CNOT
}
```

#### C++ Object

The C++ object for these gates is as one would expect. Here are some examples:


```cpp
Op x;
x.name = "x";       // "x" gate identifier
x.qubits = {qubit}; // qubit to apply gate to

Op u3;
u3.name = "u3";                  // u3 single-qubit gate identifier
u3.qubits = {qubit};             // qubit to apply gate to
u3.params_d = {theta, phi, lam}; // params for u3 gate

Op cx;
cx.name = "cx";                             // CNOT gate identifier
cx.qubits = {qubit_control, qubit_target};  // control and target qubits
```



[Back to top](#table-of-contents)


---
## Special Operations

The following are special operations which backends that support them must use the following way.


### Boolean Function

The Boolean function (`bfunc`) operation is used for classical processing and must be supported by engines that handle conditional operations. This functionality should be implemented at the `Engine` level so it can be used by multiple `State` inteferface classes.


#### JSON Schema

The JSON schema specifies that boolean operations look like

```
{
    "name": "bfunc",
    "mask": str,           // hex string for bit-mask
    "relation": str,       // boolean function for comparison (eg "==")
    "val": str,            // hex string for val comparison
    "register": list[int], // register to store value in
    "memory": list[int]    // optional memory to store value in
}
```

#### C++ Object

In C++ we define `bfunc` objects to be of the form:

```cpp
Op bfunc;
bfunc.name = "bfunc";                       // bfunc operator identifier
bfunc.params_s = {relation, mask, val };    // encoding of parameters
bfunc.registers = {registers bits, ... };   // register bits to store the output in
bfunc.memory = {memory bits, ... };         // (optional) memory bits to store output in
```

[Back to top](#table-of-contents)


### Matrix Multiplication

Direct specification of matrix operations for matrix multiplication is important for certian simulator optimizations. This object allows specification of an N-qubit matrix for multiplaction, and the qubits to apply it to. It is assumed that the identity matrix is applied to all other qubits in the system.

We allow two types of specification. Square matrices (`"mat"`), and diagonal matrices (`"dmat"`).

#### JSON Schema

Square matrices are specified as

```
{
    "name": "mat",
    "qubits": list[int],           // qubits to apply matrix to
    "params": list[complex],       // matrix to apply to qubits,
}
```

Diagonal matrices are specified by their diagonal

```
{
    "name": "dmat",
    "qubits": list[int],     // qubits to apply matrix to
    "params": list[complex], // matrix diagonal to apply to qubits,
}
```

At a JSON level when these operations are specified in a circuit, they should be enforced to be unitary matrices.

#### C++ Object

Implementing these in C++ we have that square matrices are given as:

```cpp
Op mat;
mat.name = "mat";
mat.qubits = {qubits, ...};
mat.params_m = {matrix};
```

Diagonal matrices are given as:

```cpp
Op dmat;
dmat.name = "dmat";
damt.qubits = {qubits, ...};
dmat.params_z = matrix_diagonal;
```

Note that these operations may be used to store non-unitary operations for implementing certain operations internally for simulator optimizations.

[Back to top](#table-of-contents)


### Snapshot

Snapshot is a command which records the current state of the simulator in a specified data format. It can be indexed by a key (the `slot_key`) for labelled storage, and optionally 1 or more snapshot types can be specified. Snapshots are assumed to apply to *all* qubits, and cannot be defined for subsets of qubits. 

#### JSON Schema

```
{
    "name": "#snapshot",
    "params": ["slot_key", "snapshot_type"]
}
```

Allowed snapshot types depend on the given state interface, but all should in principle support a `"default"` option (which is up to the engine/backend to determine). 


#### C++ Object

```cpp
Op.name = "#snapshot";
Op.params_s = {"slot_key", "snapshot_type1", "snapshot_type2", ...};
```

Snapshots are handled by serializing of data structures to JSON, so the state interface shouold have a JSON conversion method defined to support snapshotting.

[Back to top](#table-of-contents)


---
## Observables Operations

Calculation of expectation values of observables are supported by certain backend configurations. Hence we need a way to specify an observable. 

For an *n*-qubit system in state $|\psi\rangle$, and an *m*-qubit observable $\mathcal{0}$, where *m* is a subset of the total *n* qubits, the expectation value of $\mathcal{0}$ is given by $\langle\mathcal{O}\rangle = \text{Tr}[\mathcal{O} \rho]$ where the $m$-qubit density matrix $\rho$ is defined by $\rho = \text{Tr}_{m \not\in n}[|\psi\rangle\langle\psi|]$.

If our simulation has multiple shots (for noise instances), these values are typically reported averaged over all shots (conditional on any memory qubit registers).

For optimizations and supports observable can be broken up into 3 types:

1. Matrix observables (`"obs_mat"`): $\mathcal{O} = \sum_{i,j} m_{i,j} |i\rangle\langle j|$. Special cases for matrix observables are:
    * Diagonal matrices: $\mathcal{O} = \sum_{i} d_{i} |i\rangle\langle i|$
    * Projectors: $\mathcal{O} = |\phi\rangle\langle \phi|$
2. Pauli matrices (`"obs_pauli"`): $\mathcal{O} = \sum_{j} m_{j} P_j$
3. Measurement observables (`"obs_measure"`): $\mathcal{O} = \bigotimes_{j}|i_j\rangle\langle i_j|$

When specifying observables we allow each one to specify a list of observablse acting on the same set of qubits (rather than specifying a separate op for each one).


### Matrix Observables

The most general type of observable is an *n*-qubit matrix. This observable operator allows the specification of a length *m* list *n*-qubit matrices $\{\mathcal{O}_k : k\in[1,m]\}$ acting on the same qubits, and should return a length *m* list of complex numbers $\{z_k : k\in[1, m]\}$ where $z_k = \langle \mathcal{O}_k \rangle$.

We allow two special types of matrix observables that can be stored in more compact fasion. These are diagonal matrices $\mathcal{O} = \sum_{i} d_{i} |i\rangle\langle i|$ and rank-1 projectors $\mathcal{O}_k = |v_k\rangle\langle v_k|$. 

We store these matrices as follows:

1. General matrix: *N x N* complex-matrix
2. Diagonal matrices: *1 x N* row-vector matrix
3. Projector: *N x 1* column-vector matrix

We also include an optional string specifying the type of matrix `"mat"`, `"dmat"`, `"vec"` respectivley.

#### JSON Schema

For general matrix observables we may specify them as:

```
{
    "name": "obs_mat",               // general matrix observable
    "qubits": list[int],             // qubits to apply matrix to
    
    "params": list[complex_matrix],  // list of complex matrices
    "string_params": list[string]    // the type of matrix
}
```

#### C++ Objects

For general matrices we have

```cpp
Op obs_mat;
obs_mat.name = "obs_mat";        // matrix observable identitier
obs_mat.qubits = {qubits, ...};  // qubits that observables apply to
obs_mat.params_m = {mat1, ...};  // matrix 
observables
obs_mat.params_s = {type1, ...}; // matrix types
```

[Back to top](#table-of-contents)


### Pauli Matrix Observables

The third special case of general matrix observables are Pauli-matrix observables: $\mathcal{O}_k = \bigotimes_{j=1}^n P_j$ where $P_j \in \{I, X, Y, Z\}$. For these operators we store them as strings containing `"I", "X", "Y", "Z"`. We allow an optional field coeffs which may be used to combine a Pauli operators. For example `"params": ["ZI", "IZ"], "coeffs": [[1, 0], [-1, 0]]` would return the expectation value of $\langle ZI - IZ \rangle$.

#### JSON Schema

For Pauli's we have

```
{
    "name": "obs_pauli",     // general matrix observable
    "qubits": list[int],     // qubits to apply matrix to
    "params": list[string],  // list of projector vectors
    "coeffs": list[complex]  // (optional) coefficent to combine paulis with
}
```

#### C++ Objects

For Pauli's we have
```cpp
Op obs_pauli;
obs_pauli.name = "obs_pauli";           // projector observable identitier
obs_pauli.qubits = {qubits, ...};       // qubits that observables apply to
obs_pauli.params_s = {pauli_str1, ...}; // Pauli strings
obs_pauli.params_z = {coeff1, ...};     // (optional) coefficients for combining terms
```

Note that in the cause of Pauli's if coffiecients are used then they must be the same number as there are Pauli strings.

[Back to top](#table-of-contents)


### Measurement Observables

Another special case of observables is measurement observables. These operators return a list of $2^n$ soutcome probabilities for measurement of the given *n*-qubits in the *Z*-basis. Unlike other observables these do not require the specification of any operators as they are fixed for the given qubits. For example for two qubits `{q0, q1}` it will return the list `{P(q1=0, q0=0), P(q1=0, q0=1), P(q1=1, q0=0), P(q1=1, q0=1)}`

Note that these probabilities are computed without actually performing the measurement, unlike the `"measure"` operation it leaves the state unchanged.


#### JSON Schema

```
{
    "name": "obs_measure", // measure observable
    "qubits": list[int],   // qubits to compute measurement probabilities
}
```

#### C++ Objects

For Measurement observables we have

```cpp
Op obs_meas;
obs_meas.name = "obs_measure";   // measure observable
obs_meas.qubits = {qubits, ...}; // qubits to compute measurement probabilities
```

[Back to top](#table-of-contents)


---
## Noise Operations

We want to have the optiona of implementing noise by inserting noise operations into a sequence of operations for a given circuit.

There are two types of noise process:
1. Noise that is *independent* of the current state of the system (*[Unitary](#unitary-noise), [Reset](#reset-noise)*)
2. Noise that is *dependent* on the current state of the system (*[Readout](#readout-noise), [Kraus](#kraus-noise)*)

The *Type 1* noises can be inserted into a sequence of operations by *sampling* a given realization of the noise from a noise model specifies the distribution for possible errors for the input operations.  This may lead to a single circuit operation sequence being modified into many different sequences for each realization of noise. Note that we do not need additional operator types to describe these noise processes --- the operators inserted to realize the noise are gates, matrix multiplication, and reset ops.

**Open Question:** *At what level of abstraction should this noise sampling be handled -- the state, engine, or controller?*

The *Type 2* noise processes cannot be sampled to choose a realization independent of the state of the system. This is because the probabilities for the classical bit-flip error for readout error, and the probabilities of choosing an individual operator to apply for Kraus error, depend on the current state classical or quantum state respectively.  Hence, we need to define new types of operators for these noise processes which can be used by the Engine/State to sample from the possible noise realizations conditional on the current simulation state.


### Readout Noise

The readout noise operation is an operation on classical bits (like the [boolean function](#boolean-function)). It is used to flip the recorded bit-values from a measurement where the bit-flip probability is conditional on the true mesaurement value. Since sampling the bit-flip probability depends on the measurement outcome this must be handled at a lower level than the controller. While we define a possible probably shouldn't be actually used in JSON directly but if so this is a schema one could have.

#### JSON Schema

This operation could be serialized in JSON as follows:

```
{
    "name": "roerror",
    "memory": list[int]   // (optional) classical bits to store result in
    "register": list[int] // (optional) for conditionals
    "params": matrix      // assignment fidelity matrix
}
```

#### C++ Object

Implementing these in C++ we have

```cpp
Op roerr;
roerr.name = "roerror";                // readout error identifier
roerr.memory = {membits, ...};         // memory bits to apply noise to
roerr.registers = {regbits, ...};      // register bits to apply noise to
roerr.params_v = {assignment_fid_columns, ...}; // assignment fidelity matrix
```

**Open Question:** *Should we store the assignment fidelity as a column-vectorized matrix in params_d, or cast from complex back to reals when converting columns to distributions?*

Since this is a classical bit error we don't need the qubits specified in the operator, *however*, the specific matrix for assignment fidelities inserted in the operation typically will be dependent on the qubits that were measured. Also note that the assignment fidelity matrix is a probability matrix so all entries should be between [0, 1] and the columns should each sum to 1.

[Back to top](#table-of-contents)


### Kraus Noise

The Kraus operation is used for noise simulations. We define it since sampling a Kraus noise operator depends on the current state value of the backend and must be handled at a lower level than the controller. While we define a possible probably shouldn't be actually used in JSON directly but if so this is a schema one could have.

We note that kraus sets defined this way should be CPTP (this should be checked at generation time, not when it is applied by the backend.)

#### JSON Schema

While this operation could be serialized in JSON as follows:

```
{
    "name": "kraus",
    "qubits": list[int],                  // qubits to apply noise to
    "params": list[vectorized_matrices],  // list of Kraus operators.
}
```

#### C++ Object

Implementing these in C++ we have

```cpp
Op kraus;
kraus.name = "kraus";                // kraus identifier
kraus.qubits = {qubits, ...};        // qubits to apply operators to
kraus.params_m = {kraus_mats, ...};  // vector of vectorizied kraus operators
```

[Back to top](#table-of-contents)


### Unitary Noise

Unitary noise does not require any special operations. It can always be expressed as one or more [gate](#gates) operations or [matrix](#matrix-multiplication) operations.

Note, However, that because unitary noise may involve sampling the unitary to be applied from a distribution supplied by the noise model, the operation (or sequence of operations) inserted into a circuit will be for a single *realization* of the noise selected by the sampling process. 

[Back to top](#table-of-contents)


### Reset Noise

Reset noise, like unitary noise, does not require any special operations. It can be expressed as a combination of a [reset](#reset) operation specifying the state to be reset to, and optionally pre- and post-rotation [unitary](#unitary-noise) operations.

As with unitary noise, the specific choice for the state being reset to, and the choice of pre- and/or post-rotation unitary operators involves sampling from their respective distributions to choose a particular realization.

[Back to top](#table-of-contents)