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


Changed
-------


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
