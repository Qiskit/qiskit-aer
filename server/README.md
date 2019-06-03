# Worker Server

## How to install
1. Copy the simulator binary file to the local directory
``` 
cp qasm_simulator qiskit-aer-internal/server
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

![Flow](https://github.ibm.com/HITOMI/qiskit-aer-internal/blob/server/server/worker/Flow.png)

## How to use
## How to use
1. Install ibmq provider
```
# pip install qiskit-ibmq-provider
```

2. Copy aercloud provider to qiskit-terra installation dir manually 
```
# cp -rf qiskit-aer-internal/providers/aercloud <Python Installation Dir>/pythonx.x/site-packages/qiskit/providers
```

3. Add `from qiskit.providers.aercloud import AerCloud` to `__init__.py` file
```
# vim <Python Installation Dir>/pythonx.x/site-packages/qiskit/__init__.py

from qiskit.providers.aercloud import AerCloud 
```

4. Call AerCloud.get_backend with "qasm_simulator"

```
from qiskit import execute, AerCloud

bkend = AerCloud.get_backend("qasm_simulator", http_hosts=["http://localhost:5000"])
job = execute([circ], backend=bkend, shots=10, noise_model=noise_model)
```
