Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_.

  **Types of changes:**

  - **Added**: for new features.
  - **Changed**: for changes in existing functionality.
  - **Deprecated**: for soon-to-be removed features.
  - **Removed**: for now removed features.
  - **Fixed**: for any bug fixes.
  - **Security**: in case of vulnerabilities.


`UNRELEASED`_
=============

Added
-----
- Add support for labelled gates in noise models (#175).
- Improve efficiency of parallelization with max_memory_mb a new parameter of backend_opts (#61)
- Add optimized mcx, mcy, mcz, mcu1, mcu2, mcu3, gates to QubitVector (#124)
- Add optimized controlled-swap gate to QubitVector
- Add gate-fusion optimization for QasmContoroller, which is enabled by setting fusion_enable=true (#136)

Changed
-------
- Add basis_gates kwarg to NoiseModel init (#175).
- Depreciated "initial_statevector" backend option for QasmSimulator and StatevectorSimulator (#185)
- Rename "chop_threshold" backend option to "zero_threshold" and change default value to 1e-10 (#185).
- Add an optional parameter to `NoiseModel.as_dict()` for returning dictionaries that can be
  serialized using the standard `json` library directly. (#165)

Removed
-------


Fixed
-----

`0.1.1`_ - 2019-01-24
=====================

Added
-----
- Adds version information when using the standalone simulator (#36)
- Adds a Clifford stabilizer simulation method to the QasmSimulator (#13)
- Improve Circuit and NoiseModel instructions checking (#31)
- Add reset_error function to Noise models (#34)
- Improve error reporting at installation time (#29)
- Validate n_qubits before execution (#24)
- Add qobj method to AerJob (#19)

Removed
-------
- Reference model tests removed from the codebase (#27)

Fixed
-----
- Fix Contributing guide (#33)
- Fix an import in Terra integration tests (#33)
- Fix non-OpenMP builds (#19)



`0.1.0`_ - 2018-12-19
=====================

Added
-----
- QASM Simulator
- Statevector Simulator
- Unitary Simulator
- Noise models
- Terra integration
- Standalone Simulators support


.. _UNRELEASED: https://github.com/Qiskit/qiskit-aer/compare/0.1.1...HEAD
.. _0.1.1: https://github.com/Qiskit/qiskit-aer/compare/0.1.0...0.1.1
.. _0.1.0: https://github.com/Qiskit/qiskit-aer/compare/0.0.0...0.1.0

.. _Keep a Changelog: http://keepachangelog.com/en/1.0.0/
