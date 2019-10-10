# Connector for Distributed Environment

Aer provider (qiskit/providers) includes the connector for the distributed environment. This connector can execute the remote simulators on multiple nodes and collect the results. 

## SSH Connector

SSH connector copies qobj file command and invokes aer simulator on remote node by ssh command. 

### Setup Remote Node

1. Setup ssh key to connect the remote node without password. 

2. Create working directory on the remote node
```
% mkdir ssh_exe
```

3. Create the directory to store qobj files on the remote node
```
% mkdir ssh_exe/qobj
```

4. Put the simulator binary in the working directory on the remote node
```
% cp qasm_simulator ssh_exe

```

5. Put the configuration json about the simulator information on the remote node

```
% cp conf.json ssh_exe
```

Example of conf.json
```
{
 "allow_q_object": true,
 "backend_name": "remote_qasm_simulator",
 "backend_version": "0.1.547",
 "n_qubits": 32,
 "simulator": true,
 "conditional": true,
 "open_pulse": false,
 "memory": false,
 "local": false,
 "coupling_map": null,
 "max_shots": 8192,
 "max_experiments": 1024,
 "GPU": true,
 "basis_gates": [
  "u1",
  "u2",
  "u3",
  "cx"
 ],
 "gates": [
  {
   "name": "u1",
   "parameters": [
    "lambda"
   ],
   "qasm_def": "gate u1(lambda) q { U(0,0,lambda) q; }"
     },
  {
   "name": "u2",
   "parameters": [
    "phi",
    "lambda"
   ],
   "qasm_def": "gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }"
  },
  {
   "name": "u3",
   "parameters": [
    "theta",
    "phi",
    "lambda"
   ],
   "qasm_def": "gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }"
  },
  {
   "name": "cx",
   "parameters": [
    "c",
    "t"
   ],
   "qasm_def": "gate cx c,t { CX c,t; }"
  }
 ]
}
```

### Example Code

Define `RemoteConfig` and call `Aer.setup_remote()`.

```
remote_config =  [ {
"host" : "host1", 
"ssh_key"  : "/home/user/.ssh/ssh-key", 
"ssh_user" : "user", 
"qobj_path" : "/home/user/ssh_exe/qobj", 
"conf_path" : "/home/user/ssh_exe/conf.json", 
"run_command" : "/home/user/ssh_exe/qasm_simulator"}, 
{ "host" : "host2", 
"ssh_key"  : "/home/user/.ssh/yakumo-kube", 
"ssh_user" : "user", 
"qobj_path" : "/home/user/ssh_exe/qobj", 
"conf_path" : "/home/user/ssh_exe/conf.json", 
"run_command" : "/home/user/ssh_exe/qasm_simulator"}
]


config = RemoteConfig(remote_config)
Aer.setup_remote(config, name="ssh")
bkend = Aer.get_backend("remote_qasm_simulator")
job = execute([qft3], backend=bkend)
```

### Remote Config Membper
- host : hostname for the remote node
- ssh_key : ssh key to connect the remote node
- ssh_user : ssh user to connect the remote node
- qobj_path : directory path in the remote to store qobj files
- conf_path : file path for configuration file in the remote node
- run_command : invoke command of Aer simulator in the remote node
