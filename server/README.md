
# Worker Server

## How to install
1. Copy the simulator binary file to the local directory
``` 
cp qasm_simulator qiskit-aer/server
```

2. Update a configuration file (server.ini)
```
[flask]
port = <Port Number of Flask Server>

[simulator]
simulator_command = <Command to Execute GPU Simulator>
qiskit_qobj_path = <Path to the Directory to Locate Qobj Files>
qiskit_job_path = <Path to the Directory to Locate Job Files>
qiskit_result_path = <Path to the Directory to Locate Result Files>
thread_num = <The Number of Permits to Execute Simulations>
```
3. Install flask  
```
# pip install flask
```
4. Run the worker server
```
# python server.py
```
## Execution Flow

![Flow](https://github.com/hitomitak/qiskit-aer/blob/distribute/server/Flow.png)

## How to use
1. Uninstall qiskit-aer
```
# pip uninstall qiskit-aer
```

2. Re-install qiksit-aer from source ( https://github.com/Qiskit/qiskit-aer/blob/master/.github/CONTRIBUTING.md )

4. Call Aer.get_backend with "http_host" 

5. If you want to enable GPU calculation, please add "GPU" options to execute function
```
from qiskit import execute, Aer

bkend = Aer.get_backend("qasm_simulator", http_hosts=["http://localhost:5000", "http://localhost:5001"])
job = execute([circ], backend=bkend, shots=10, noise_model=noise_model, run_config={"GPU":True})
```
