# Compiling stand-alone binaries of simulators

This folder contains makefiles to build standalone executables for simulators. It is used for testing but the final build method of AER should use direct linking of the C++ libraries in python using Cython or a similar method.


### Build 

To build first make sure that all [build dependencies](#build-dependencies) for the OS have been installed. Following that build the simulator using `make` as by executing the following commands from the *qiskit-aer* directory

```bash
cd legacy-build
make clean; make
```

Currently it builds the simulator at the following location:

```bash
qiskit-aer/out/aer-simulator/aer_simulator
```

### Running

To run a Qobj on the simulator from the *qiskit-aer* directory run

```bash
out/aer-simulator/aer_simulator qobj_file.json
```

Several example Qobj files are located in the `examples/qobj` directory.

### Build Dependencies

#### MacOS

To build on MacOS with OpenMP support requries XCode developer tools and LLVM OpenMP. The LLVM OpenMP library can be installed using the [Homebrew](https://brew.sh/) package manager. This can be installed using the following commands:

```bash
xcode-select --install
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install libomp
```

#### Ubuntu Linux

Building on Ubuntu requires the installation of build essential package along with BLAS and LAPACK. This can be installed using the following commands:

```bash
apt-get update
apt-get -y install build-essential libblas-dev liblapack-dev
```

#### Redhat Linux

Building on Redhat requires installation of a recent developer toolset to enable a C++11 compatible GCC compiler, it also requires BLAS and LAPACK packages to be installed. This can be installed using the following commands:

```bash
yum update
yum -y install devtoolset-6 blas blas-devel lapack lapack-devel
scl enable devtoolset-6 bash
```

