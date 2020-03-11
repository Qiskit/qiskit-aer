# Openpulse

This simulates job using the open pulse format (see the spec).

## Example
The Hamiltonian `H=-w_0 * \sigma_z/2  + w_1 * cos(w t + \phi) * \sigma_x/2` may be specified as:
```
hamiltonian = {}
hamiltonian['h_str'] = []

#Q0 terms
hamiltonian['h_str'].append('-0.5*w0*Z0')
hamiltonian['h_str'].append('0.5*w1*X0||D0')

hamiltonian['vars'] = {'w0': (add w0 val here), 'w1': (add w1 value here)}

# set the qubit dimension to 2
hamiltonian['qub'] = {'0': 2}
```
This Hamiltonian has a closed form in the rotating frame specified in the solving section.

Note: Variable names must be lowercase (uppercase is reserved for operators).

## Hamiltonian

One of the main components of the open pulse simulator is the specification
of the Hamiltonian. The Hamiltonian dictates how the simulation will evolve
the dynamics based on the pulse schedule.

There are two allowed components to a simulation, *qubits* and *oscillators*.
*Qubits* are the objects that will be measured at the end of the simulation.
The operators for qubits are defined as Pauli's but if the number of levels
is defined to be greater than 2 these will be internally converted to
creation and annhilation operators.

The Hamiltonian is a dictionary comprised of
- `h_str`: Definition of the Hamiltonian in terms of operators, drives and
coefficients
- `vars`: Numeric values for the variables in the Hamiltonian
- `qubs`: Dictionary indicating the number of levels for each qubit to
use in the simulation
- `osc`: Dictionary indicating the number of levels for each oscillator to
use in the simulation

There must be qubits, but there does not need to be oscillators. Measurements
are given by the qubit state; the measurement process is not simulated.

### `h_str` Syntax

The `h_str` is a list of strings which indicate various terms in the
Hamiltonian. These can have the form

`<op>||<ch>` or `<op>`

where `<op>` is a string form of an operator with coefficients and `<ch>`
indicates one of the time-dependent drive channels (e.g. `D2` or `U10`).
Additionally, there is a sum form
`_SUM[i,n1,n2,F[{i}]]` where {i} is replaced in each loop by the value.

Available operators are: 
{'X': sigmax, 'Y': sigmay, 'Z': sigmaz,
 'Sp': creation (sigma plus), 'Sm': destruction (sigma minus), 'I': identity,
 'O': number, 'P': projection, 'A': destruction, 'C': creation, 'N': number op}
 
 The following functions are also available: 
 {'cos': cos, 'sin': sin, 'exp': exp,
  'sqrt': sqrt, 'conj': complex conjugate, 'dag': dagger (Hermitian conjugate)}
 
## Solving

The solver takes the Hamiltonian (`H(t=0)`) and sets all drive/control channels to zero. 
Consider `H_d` to be the diagonal elements of `H(t=0)`, then the transformation applied to the statevector `\psi` is
`U=e^{-i H_d t/hbar}` (`\psi \rightarrow U \psi). For all drive/control channels, the LO is applied so
`d(t) -> D(t)e^{-i w t}`. The LO frequency `w` may be set in the pulse. If the drive is associated with some operator *B* then
the upper triangular part of *B* is multiplied by `d(t)` and the lower triangular part
by `d*(t)`. This ensures the Hamiltonian is Hermitian and also that in the transformed
frame that a resonant pulse is close to DC. 

### Measurement

The measurement operators are the projections onto the 1 excitation subsystem for qubit `l`
where qubit `l` is defined by diagonalizing `H(t=0)` (i.e. the dressed basis). 

There are three measurement levels that return the data.
Measurement level `0` gives the raw data.
Measurement level `1` gives complex numbers (IQ values).
Measurement level `2` gives the discriminated states, `|0>` and `|1>`.

