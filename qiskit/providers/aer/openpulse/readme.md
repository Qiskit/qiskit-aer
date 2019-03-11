# Openpulse

This simulates job using the open pulse format (see the spec).

## Example

XXX

## Hamiltonian

One of the main components of the open pulse simulator is the specification
of the Hamiltonian. The Hamiltonian dictates how the simulation will evolve
the dynamics based on the pulse schedule.

There are two allowed components to a simulation, *qubits* and *oscillators*.
*Qubits* are the objects that will be measured at the end of the simulation.
The operators for qubits are defined as Pauli's but if the number of levels
is defined by be greater than 2 these will be internally converted to
creation and annhilation operators.

The Hamiltonian is a dictionary comparised of
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

## Solving

The solver takes the Hamiltonian (`H(t=0)`) and sets all drive/control channels to zero. 
Consider `H_d` to be the diagonal elements of `H(t=0)`, then the transformation applied is
`U=e^{-i H_d t/hbar}`. For all drive/control channels, the LO is applied so
`d(t) -> D(t)e^{-i w t}`. If the drive is associated with some operator *B* then
the upper triangular part of *B* is multiplied by `d(t)` and the lower triangular part
by `d*(t)`. This ensures the Hamiltonian is Hermitian and also that in the transformed
frame that a resonant pulse is close to DC. 

### Measurement

The measurement operators are the projections onto the 1 excitation subsystem for qubit `l`
where qubit `l` is defined by diagonalizing `H(t=0)` (i.e. the dressed basis). 

