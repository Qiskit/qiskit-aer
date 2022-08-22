The main goal of the benchmarking framework is to detect regressions during development, but one can run benchmarks at any specific commit just to see how it performs, this can be a very useful tool for Aer developers so they can make sure their changes don't introduce important performance regressions.

Our benchmarking framework is based on [Airspeed Velocity](https://asv.readthedocs.io/).
We have only implemented benchmarks for the Qiskit Addon, not the standalone mode.

# Where are the benchmarks
All the benchmarks are under the `test/benchmark` directory.
There you'll find python files with names of simulation methods, such as `densitymatrix_cpu` and `statevector_gpu`.
Each of them runs Quantum Volume, QFT, and RealAmplitudes circuits with a specific simulation method and device with and without noise models.

# How to run the benchmarks
All prerequisites for building the project need to be installed in the system, take a look at the [CONTRIBUTING guide](.github/CONTRIBUTING.md) if you don't have them already installed.

Install Airspeed Velocity (`ASV`):
```
$ pip install asv
```

Move to the `test` directory:
```
$ cd test
```

And run `asv` using the correct configuration file, depending on what O.S. you are executing them:
Linux:
```
$ asv run --config asv.linux.conf.json
```

MacOS:
```
$ asv run --config asv.macos.conf.json
```

NOTE: We only support Linux and MacOS at the moment

Depending on your system, benchmarks will take a while to complete.
After the completion of the tests, you will see the results with a format similar like this:
```
· Creating environments
· Discovering benchmarks
·· Uninstalling from conda-py3.8
·· Installing 71d55588 <main> into conda-py3.8
· Running 58 total benchmarks (1 commits * 1 environments * 58 benchmarks)
[  0.00%] · For qiskit-aer commit 71d55588 <main>:
[  0.00%] ·· Benchmarking conda-py3.8
[  1.72%] ··· default_simulator.Benchmark.track_qft                           ok
[  1.72%] ··· ======== =====================
               qubits                       
              -------- ---------------------
                 5      0.01046895980834961 
                 15     0.07767915725708008 
                 25      0.7862622737884521 
              ======== =====================

[  3.45%] ··· default_simulator.Benchmark.track_qv                            ok
[  3.45%] ··· ======== =====================
               qubits                       
              -------- ---------------------
                 5      0.01030588150024414 
                 15     0.04601025581359863 
                 25      2.6632273197174072 
              ======== =====================
```
