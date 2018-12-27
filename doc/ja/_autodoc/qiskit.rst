qiskit package
==============


.. automodule:: qiskit
    
    
    Submodules
    ----------

    .. toctree::
       :maxdepth: 1
   


       
       qiskit.exceptions
       qiskit.unroll
       qiskit.version

    Subpackages
    -----------

    .. toctree::
       :maxdepth: 1

       qiskit.circuit
       qiskit.converters
       qiskit.dagcircuit
       qiskit.extensions
       qiskit.mapper
       qiskit.providers
       qiskit.qasm
       qiskit.qobj
       qiskit.quantum_info
       qiskit.result
       qiskit.transpiler
       qiskit.unrollers
       qiskit.validation
       qiskit.wrapper

    Exceptions
    ----------


    .. list-table::
    
       * - :exc:`QISKitError <qiskit.exceptions.QISKitError>`
         - Old Base class for errors raised by the Qiskit for backwards compat only, not for use.
       * - :exc:`QiskitError <qiskit.exceptions.QiskitError>`
         - Base class for errors raised by the Qiskit.
    

    Classes
    -------


    .. list-table::
    
       * - :class:`ClassicalRegister <qiskit.circuit.classicalregister.ClassicalRegister>`
         - Implement a classical register.
       * - :class:`QuantumCircuit <qiskit.circuit.quantumcircuit.QuantumCircuit>`
         - Quantum circuit.
       * - :class:`QuantumRegister <qiskit.circuit.quantumregister.QuantumRegister>`
         - Implement a quantum register.
    



    .. _qiskit_top_level_functions:


    Functions
    ---------


    .. list-table::
    
       * - :func:`compile <qiskit.tools.compiler.compile>`
         - Compile a list of circuits into a qobj.
       * - :func:`execute <qiskit.tools.compiler.execute>`
         - Executes a set of circuits.
       * - :func:`load_qasm_file <qiskit.wrapper._wrapper.load_qasm_file>`
         - Construct a quantum circuit from a qasm representation (file).
       * - :func:`load_qasm_string <qiskit.wrapper._wrapper.load_qasm_string>`
         - Construct a quantum circuit from a qasm representation (string).
    