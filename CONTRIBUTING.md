# Contributing

First read the overall project contributing guidelines. These are all
included in the qiskit documentation:

https://qiskit.org/documentation/contributing_to_qiskit.html

## Contributing to Qiskit Aer

In addition to the general guidelines there are specific details for
contributing to aer, these are documented below.

### Pull request checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the code style of the project and successfully
   passes the tests. For convenience, you can execute `tox` locally,
   which will run these checks and report any issues.
2. The documentation has been updated accordingly. In particular, if a
   function or class has been modified during the PR, please update the
   *docstring* accordingly.
3. If it makes sense for your change that you have added new tests that
   cover the changes.
4. Ensure that if your change has an end user facing impact (new feature,
   deprecation, removal etc) that you have added a reno release note for that
   change and that the PR is tagged for the changelog.

### Changelog generation

The changelog is automatically generated as part of the release process
automation. This works through a combination of the git log and the pull
request. When a release is tagged and pushed to github the release automation
bot looks at all commit messages from the git log for the release. It takes the
PR numbers from the git log (assuming a squash merge) and checks if that PR had
a `Changelog:` label on it. If there is a label it will add the git commit
message summary line from the git log for the release to the changelog.

If there are multiple `Changelog:` tags on a PR the git commit message summary
line from the git log will be used for each changelog category tagged.

The current categories for each label are as follows:

| PR Label               | Changelog Category |
| -----------------------|--------------------|
| Changelog: Deprecation | Deprecated         |
| Changelog: New Feature | Added              |
| Changelog: API Change  | Changed            |
| Changelog: Removal     | Removed            |
| Changelog: Bugfix      | Fixed              |

### Release Notes

When making any end user facing changes in a contribution we have to make sure
we document that when we release a new version of qiskit-aer. The expectation
is that if your code contribution has user facing changes that you will write
the release documentation for these changes. This documentation must explain
what was changed, why it was changed, and how users can either use or adapt
to the change. The idea behind release documentation is that when a naive
user with limited internal knowledge of the project is upgrading from the
previous release to the new one, they should be able to read the release notes,
understand if they need to update their program which uses qiskit, and how they
would go about doing that. It ideally should explain why they need to make
this change too, to provide the necessary context.

To make sure we don't forget a release note or if the details of user facing
changes over a release cycle we require that all user facing changes include
documentation at the same time as the code. To accomplish this we use the
[reno](https://docs.openstack.org/reno/latest/) tool which enables a git based
workflow for writing and compiling release notes.

#### Adding a new release note

Making a new release note is quite straightforward. Ensure that you have reno
installed with::

    pip install -U reno

Once you have reno installed you can make a new release note by running in
your local repository checkout's root::

    reno new short-description-string

where short-description-string is a brief string (with no spaces) that describes
what's in the release note. This will become the prefix for the release note
file. Once that is run it will create a new yaml file in releasenotes/notes.
Then open that yaml file in a text editor and write the release note. The basic
structure of a release note is restructured text in yaml lists under category
keys. You add individual items under each category and they will be grouped
automatically by release when the release notes are compiled. A single file
can have as many entries in it as needed, but to avoid potential conflicts
you'll want to create a new file for each pull request that has user facing
changes. When you open the newly created file it will be a full template of
the different categories with a description of a category as a single entry
in each category. You'll want to delete all the sections you aren't using and
update the contents for those you are. For example, the end result should
look something like::

```yaml
features:
  - |
    Introduced a new feature foo, that adds support for doing something to
    ``QuantumCircuit`` objects. It can be used by using the foo function,
    for example::

      from qiskit import foo
      from qiskit import QuantumCircuit
      foo(QuantumCircuit())

  - |
    The ``qiskit.QuantumCircuit`` module has a new method ``foo()``. This is
    the equivalent of calling the ``qiskit.foo()`` to do something to your
    QuantumCircuit. This is the equivalent of running ``qiskit.foo()`` on
    your circuit, but provides the convenience of running it natively on
    an object. For example::

      from qiskit import QuantumCircuit

      circ = QuantumCircuit()
      circ.foo()

deprecations:
  - |
    The ``qiskit.bar`` module has been deprecated and will be removed in a
    future release. Its sole function, ``foobar()`` has been superseded by the
    ``qiskit.foo()`` function which provides similar functionality but with
    more accurate results and better performance. You should update your calls
    ``qiskit.bar.foobar()`` calls to ``qiskit.foo()``.
```

You can also look at other release notes for other examples.

You can use any restructured text feature in them (code sections, tables,
enumerated lists, bulleted list, etc) to express what is being changed as
needed. In general you want the release notes to include as much detail as
needed so that users will understand what has changed, why it changed, and how
they'll have to update their code.

After you've finished writing your release notes you'll want to add the note
file to your commit with `git add` and commit them to your PR branch to make
sure they're included with the code in your PR.

##### Linking to issues

If you need to link to an issue or other github artifact as part of the release
note this should be done using an inline link with the text being the issue
number. For example you would write a release note with a link to issue 12345
as:

```yaml
fixes:
  - |
    Fixes a race condition in the function ``foo()``. Refer to
    `#12345 <https://github.com/Qiskit/qiskit-aer/issues/12345>` for more
    details.
```

#### Generating the release notes

After release notes have been added if you want to see what the full output of
the release notes. In general the output from reno that we'll get is a rst
(ReStructuredText) file that can be compiled by
[sphinx](https://www.sphinx-doc.org/en/master/). To generate the rst file you
use the ``reno report`` command. If you want to generate the full aer release
notes for all releases (since we started using reno during 0.9) you just run::

    reno report

but you can also use the ``--version`` argument to view a single release (after
it has been tagged::

    reno report --version 0.5.0

At release time ``reno report`` is used to generate the release notes for the
release and the output will be submitted as a pull request to the documentation
repository's [release notes file](
https://github.com/Qiskit/qiskit/blob/master/docs/release_notes.rst)

#### Building release notes locally

Building The release notes are part of the standard qiskit-aer documentation
builds. To check what the rendered html output of the release notes will look
like for the current state of the repo you can run: `tox -edocs` which will
build all the documentation into `docs/_build/html` and the release notes in
particular will be located at `docs/_build/html/release_notes.html`

### Development Cycle

The development cycle for qiskit-aer is all handled in the open using
the project boards in Github for project management. We use milestones
in Github to track work for specific releases. The features or other changes
that we want to include in a release will be tagged and discussed in Github.
As we're preparing a new release we'll document what has changed since the
previous version in the release notes.

### Branches

* `master`:

The master branch is used for development of the next version of qiskit-aer.
It will be updated frequently and should not be considered stable. The API
can and will change on master as we introduce and refine new features.

* `stable/*` branches:
Branches under `stable/*` are used to maintain released versions of qiskit-aer.
It contains the version of the code corresponding to the latest release for
that minor version on pypi. For example, stable/0.4 contains the code for the
0.4.0 release on pypi. The API on these branches are stable and the only changes
merged to it are bugfixes.

### Release cycle

When it is time to release a new minor version of qiskit-aer we will:

1.  Create a new tag with the version number and push it to github
2.  Change the `master` version to the next release version.

The release automation processes will be triggered by the new tag and perform
the following steps:

1.  Create a stable branch for the new minor version from the release tag
    on the `master` branch
2.  Build and upload binary wheels to pypi
3.  Create a github release page with a generated changelog
4.  Generate a PR on the meta-repository to bump the Aer version and
    meta-package version.

The `stable/*` branches should only receive changes in the form of bug
fixes.


## Install from Source

>  Note: The following are prerequisites for all operating systems

We recommend using Python virtual environments to cleanly separate Qiskit from
other applications and improve your experience.

- The simplest way to use environments is by using *Anaconda* in a terminal
window
```
    $ conda create -y -n QiskitDevEnv python=3
    $ conda activate QiskitDevEnv
```

- Clone the `Qiskit Aer` repo via *git*.
```
    $ git clone https://github.com/Qiskit/qiskit-aer
```

- Next, install the platform-specific dependencies for your operating system [Linux](#linux-dependencies) | [macOS](#mac-dependencies) | [Windows](#win-dependencies).

- The common dependencies can then be installed via *pip*, using the
`requirements-dev.txt` file, e.g.:
```
    $ cd qiskit-aer
    $ pip install -r requirements-dev.txt
```

This will also install [**Conan**](https://conan.io/), a C/C++ package manager written in Python. This tool will handle
most of the dependencies needed by the C++ source code. Internet connection may be needed for the first build or
when dependencies are added/updated, in order to download the required packages if they are not in your **Conan** local
repository.

>  Note: Conan use can be disabled with the flag or environment variable ``DISABLE_CONAN=ON`` .
This is useful for building from source offline, or to reuse the installed package dependencies.

If we are only building the standalone version and do not want to install all Python requirements you can just install
**Conan**:

    $ pip install conan

You're now ready to build from source! Follow the instructions for your platform: [Linux](#linux-build) | [macOS](#mac-build) | [Windows](#win-build)

### Linux

Qiskit is officially supported on Red Hat, CentOS, Fedora and Ubuntu distributions, as long as you can install a GCC version that is C++14 compatible and the few dependencies we need.

#### <a name="linux-dependencies"> Dependencies </a>

To get most of the necessary compilers and libraries, install the *development environment* tools from your Linux distribution by running

CentOS/Red Hat

    $ yum groupinstall "Development Tools"

Fedora

    $ dnf install @development-tools

Ubuntu

    $ sudo apt install build-essential

Although the *BLAS* and *LAPACK* library implementations included in the
*build-essential* package are sufficient to build all of the `Aer` simulators, we
recommend using *OpenBLAS*, which you can install by running

CentOS/Red Hat

    $ yum install openblas-devel

Fedora

    $ dnf install openblas-devel

Ubuntu

    $ sudo apt install libopenblas-dev


And of course, `git` is required in order to build from repositories

CentOS/Red Hat

    $ yum install git

Fedora

    $ dnf install git

Ubuntu

    $ apt-get install git

#### <a name="linux-build"> Build </a>

There are two ways of building `Aer` simulators, depending on your goal:

1. Build a python extension that works with Terra.
2. Build a standalone executable.

**Python extension**

As any other python package, we can install from source code by just running:

    qiskit-aer$ pip install .

This will build and install `Aer` with the default options which is probably suitable for most of the users.
There's another pythonic approach to build and install software: build the wheels distributable file.

    qiskit-aer$ python ./setup.py bdist_wheel

This is also the way we will choose to change default `Aer` behavior by passing parameters to the build system.


**Advanced options**

As `Aer` is meant to be executed in many configurations and platforms, there is a complex underlying build system that offers a lot of options you can tune by setting some parameters.

We are using [*scikit-build*](https://scikit-build.readthedocs.io/en/latest/index.html) as a substitute for *setuptools*. This is
basically the glue between *setuptools* and *CMake*, so there are various
options to pass variables to *CMake*, and the underlying build system
(depending on your platform). The way to pass variables is:

    qiskit-aer$ python ./setup.py bdist_wheel [skbuild_opts] \
    [-- [cmake_opts] [-- build_tool_opts]]

where the elements within square brackets `[]` are optional, and
*`skbuild_opts`*, *`cmake_opts`*, *`build_tool_opts`* are to be replaced by
flags of your choice. A list of *CMake* options is available
[here](https://cmake.org/cmake/help/v3.6/manual/cmake.1.html#options). For
example,

    qiskit-aer$ python ./setup.py bdist_wheel --build-type=Debug -- -DCMAKE_CXX_COMPILER=g++-9 -- -j8

This is passing the `--build-type` option with `Debug` parameter to scikit-build, so we are telling it to perform a debug build. The `-DCMAKE_CXX_COMPILER=g++-9` option is being passed to `CMake` so it forces the use of `g++-9` compiler, and the `-j8` flag is telling the underlying build system, which in this case is *Makefile*, to build in parallel using 8 processes.

After this command is executed successfully, we will have a wheel package into
the `dist/` directory, so next step is installing it:

    qiskit-aer/dist$ pip install -U dist/qiskit_aer*.whl

**Standalone Executable**

If we want to build a standalone executable, we have to use *CMake* directly.
The preferred way *CMake* is meant to be used, is by setting up an "out of
source" build. So in order to build our standalone executable, we have to follow
these steps:

    qiskit-aer$ mkdir out
    qiskit-aer$ cd out
    qiskit-aer/out$ cmake ..
    qiskit-aer/out$ cmake --build . --config Release -- -j4

Once built, you will have your standalone executable into the `Release/` or
`Debug/` directory (depending on the type of building chosen with the `--config`
option):

    qiskit-aer/out$ cd Release
    qiskit-aer/out/Release/$ ls
    qasm_simulator


**Advanced options**

Because the standalone version of `Aer` doesn't need Python at all, the build system is
based on CMake, just like most of other C++ projects. So in order to pass all the different
options we have on `Aer` to CMake we use it's native mechanism:

    qiskit-aer/out$ cmake -DCMAKE_CXX_COMPILER=g++-9 -DAER_BLAS_LIB_PATH=/path/to/my/blas ..


### macOS

#### <a name="mac-dependencies"> Dependencies </a>

We recommend installing *OpenBLAS*, which is our default choice:

    $ brew install openblas

The *CMake* build system will search for other *BLAS* implementation
alternatives if *OpenBLAS* is not installed in the system.

You further need to have *Xcode Command Line Tools* installed on macOS:

    $ xcode-select --install

#### <a name="mac-build"> Build </a>

There are two ways of building `Aer` simulators, depending on your goal:

1. Build a python extension that works with Terra;
2. Build a standalone executable.

**Python extension**

As any other python package, we can install from source code by just running:

    qiskit-aer$ pip install .

This will build and install `Aer` with the default options which is probably suitable for most of the users.
There's another pythonic approach to build and install software: build the wheels distributable file.


   qiskit-aer$ python ./setup.py bdist_wheel


This is also the way we will choose to change default `Aer` behavior by passing parameters to the build system.

***Advanced options***

As `Aer` is meant to be executed in many configurations and platforms, there is a complex underlying build system that offers a lot of options you can tune by setting some parameters.

We are using [*scikit-build*](https://scikit-build.readthedocs.io/en/latest/index.html) as a substitute for *setuptools*. This is
basically the glue between *setuptools* and *CMake*, so there are various
options to pass variables to *CMake*, and the underlying build system
(depending on your platform). The way to pass variables is:

    qiskit-aer$ python ./setup.py bdist_wheel [skbuild_opts] [-- [cmake_opts] [-- build_tool_opts]]

where the elements within square brackets `[]` are optional, and
*`skbuild_opts`*, *`cmake_opts`*, *`build_tool_opts`* are to be replaced by
flags of your choice. A list of *CMake* options is available
[here](https://cmake.org/cmake/help/v3.6/manual/cmake.1.html#options). For
example,

    qiskit-aer$ python ./setup.py bdist_wheel --build-type=Debug -- -DCMAKE_CXX_COMPILER=g++-9 -- -j8

This is passing the `--build-type` option with `Debug` parameter to scikit-build, so we are telling it to perform a debug build. The `-DCMAKE_CXX_COMPILER=g++-9` option is being passed to `CMake` so it forces the use of `g++-9` compiler, and the `-j8` flag is telling the underlying build system, which in this case is *Makefile*, to build in parallel using 8 processes.

After this command is executed successfully, we will have a wheel package into
the `dist/` directory, so next step is installing it:

    qiskit-aer/dist$ pip install -U dist/qiskit_aer*.whl

**Standalone Executable**

If we want to build a standalone executable, we have to use **CMake** directly.
The preferred way **CMake** is meant to be used, is by setting up an "out of
source" build. So in order to build our standalone executable, we have to follow
these steps:

    qiskit-aer$ mkdir out
    qiskit-aer$ cd out
    qiskit-aer/out$ cmake ..
    qiskit-aer/out$ cmake --build . --config Release -- -j4

Once built, you will have your standalone executable into the `Release/` or
`Debug/` directory (depending on the type of building chosen with the `--config`
option):

    qiskit-aer/out$ cd Release
    qiskit-aer/out/Release/$ ls
    qasm_simulator

***Advanced options***

Because the standalone version of `Aer` doesn't need Python at all, the build system is
based on CMake, just like most of other C++ projects. So in order to pass all the different
options we have on `Aer` to CMake we use it's native mechanism:

    qiskit-aer/out$ cmake -DCMAKE_CXX_COMPILER=g++-9 -DAER_BLAS_LIB_PATH=/path/to/my/blas ..



### Windows

#### <a name="win-dependencies"> Dependencies </a>

On Windows, you must have *Anaconda3* installed. We recommend also installing
*Visual Studio 2017 Community Edition* or *Visual Studio 2019 Community Edition*.

>*Anaconda 3* can be installed from their web:
>https://www.anaconda.com/distribution/#download-section
>
>*Visual Studio 2017/2019 Community Edition* can be installed from:
>https://visualstudio.microsoft.com/vs/community/

Once you have *Anaconda3* and *Visual Studio Community Edition* installed, you have to open a new cmd terminal and
create an Anaconda virtual environment or activate it if you already have created one:

    > conda create -y -n QiskitDevEnv python=3
    > conda activate QiskitDevEnv
    (QiskitDevEnv) >_

We only support *Visual Studio* compilers on Windows, so if you have others installed in your machine (MinGW, TurboC)
you have to make sure that the path to the *Visual Studio* tools has precedence over others so that the build system
can get the correct one.
There's a (recommended) way to force the build system to use the one you want by using CMake `-G` parameter. Will talk
about this and other parameters later.

#### <a name="win-build"> Build </a>

**Python extension**

As any other python package, we can install from source code by just running:

    (QiskitDevEnv) qiskit-aer > pip install .

This will build and install `Aer` with the default options which is probably suitable for most of the users.
There's another pythonic approach to build and install software: build the wheels distributable file.


   (QiskitDevEnv) qiskit-aer > python ./setup.py bdist_wheel


This is also the way we will choose to change default `Aer` behavior by passing parameters to the build system.

***Advanced options***

As `Aer` is meant to be executed in many configurations and platforms, there is a complex underlying build system that offers a lot of options you can tune by setting some parameters.

We are using [*scikit-build*](https://scikit-build.readthedocs.io/en/latest/index.html) as a substitute for *setuptools*. This is
basically the glue between *setuptools* and *CMake*, so there are various
options to pass variables to *CMake*, and the underlying build system
(depending on your platform). The way to pass variables is:

    qiskit-aer > python ./setup.py bdist_wheel [skbuild_opts] [-- [cmake_opts] [-- build_tool_opts]]

where the elements within square brackets `[]` are optional, and
*`skbuild_opts`*, *`cmake_opts`*, *`build_tool_opts`* are to be replaced by
flags of your choice. A list of *CMake* options is available
[here](https://cmake.org/cmake/help/v3.6/manual/cmake.1.html#options). For
example,

    (QiskitDevEnv) qiskit-aer > python ./setup.py bdist_wheel --build-type=Debug -- -G "Visual Studio 15 2017"

This is passing the `--build-type` option with `Debug` parameter to scikit-build, so we are telling it to perform a debug build. The `-G "Visual Studio 15 2017"` option is being passed to `CMake` so it forces the use of `Visual Studio 2017` C++ compiler to drive the build.

After this command is executed successfully, we will have a wheel package into
the `dist/` directory, so next step is installing it:

    (QiskitDevEnv) qiskit-aer\dist$ pip install -U dist\qiskit_aer*.whl

**Standalone Executable**

If we want to build a standalone executable, we have to use **CMake** directly.
The preferred way **CMake** is meant to be used, is by setting up an "out of
source" build. So in order to build our standalone executable, we have to follow
these steps:

    (QiskitDevEnv) qiskit-aer> mkdir out
    (QiskitDevEnv) qiskit-aer> cd out
    (QiskitDevEnv) qiskit-aer\out> cmake ..
    (QiskitDevEnv) qiskit-aer\out> cmake --build . --config Release -- -j4

Once built, you will have your standalone executable into the `Release/` or
`Debug/` directory (depending on the type of building chosen with the `--config`
option):

    (QiskitDevEnv) qiskit-aer\out> cd Release
    (QiskitDevEnv) qiskit-aer\out\Release> dir
    qasm_simulator

***Advanced options***

Because the standalone version of `Aer` doesn't need Python at all, the build system is
based on CMake, just like most of other C++ projects. So in order to pass all the different
options we have on `Aer` to CMake we use it's native mechanism:

    (QiskitDevEnv) qiskit-aer\out> cmake -G "Visual Studio 15 2017" -DAER_BLAS_LIB_PATH=c:\path\to\my\blas ..


### Building with GPU support

Qiskit Aer can exploit GPU's horsepower to accelerate some simulations, specially the larger ones.
GPU access is supported via CUDA® (NVIDIA® chipset), so in order to build with GPU support we need
to have CUDA® >= 10.1 preinstalled. See install instructions [here](https://developer.nvidia.com/cuda-toolkit-archive)
Please note that we only support GPU acceleration on Linux platforms at the moment.

Once CUDA® is properly installed, we only need to set a flag so the build system knows what to do:

```
AER_THRUST_BACKEND=CUDA
```

For example,

    qiskit-aer$ python ./setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA

If we want to specify the CUDA® architecture instead of letting the build system
auto detect it, we can use the AER_CUDA_ARCH flag (can also be set as an ENV variable
with the same name, although the flag takes precedence). For example:

    qiskit-aer$ python ./setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA -DAER_CUDA_ARCH="5.2"

or

    qiskit-aer$ export AER_CUDA_ARCH="5.2"
    qiskit-aer$ python ./setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA

This will reduce the amount of compilation time when, for example, the architecture auto detection
fails and the build system compiles all common architectures.

Few notes on GPU builds:
1. Building takes considerable more time than non-GPU build, so be patient :)
2. CUDA® >= 10.1 imposes the restriction of building with g++ version not newer than 8
3. We don't need NVIDIA® drivers for building, but we need them for running simulations
4. Only Linux platforms are supported

### Building a statically linked wheel

If you encounter an error similar to the following, you may are likely in the need of compiling a
statically linked wheel.
```
    ImportError: libopenblas.so.0: cannot open shared object file: No such file or directory
```
However, depending on your setup this can proof difficult at times.
Thus, here we present instructions which are known to work under Linux.

In general, the workflow is:
1. Compile a wheel
```
    qiskit-aer$ python ./setup.py bdist_wheel
```
2. Repair it with [auditwheel](https://github.com/pypa/auditwheel)
```
    qiskit-aer$ auditwheel repair dist/qiskit_aer*.whl
```
> `auditwheel` vendors the shared libraries into the binary to make it fully self-contained.

The command above will attempt to repair the wheel for a `manylinux*` platform and will store it
under `wheelhouse/` from where you can install it.

It may happen that you encounter the following error:
```
    auditwheel: error: cannot repair "qiskit_aer-0.8.0-cp36-cp36m-linux_x86_64.whl" to "manylinux1_x86_64" ABI because of the presence of too-recent versioned symbols. You'll need to compile the wheel on an older toolchain.
```
This means that your toolchain uses later versions of system libraries than are allowed by the
`manylinux*` platform specification (see also [1], [2] and [3]).
If you do not need your wheel to support the `manylinux*` platform you can resolve this issue by
limiting the compatibility of your wheel to your specific platform.
You can find out which platform this is through
```
    qiskit-aer$ auditwheel show dist/qiskit_aer*.whl
```
This will list the _platform tag_ (e.g. `linux_x86_64`).
You can then repair the wheel for this specific platform using:
```
    qiskit-aer$ auditwheel repair --plat linux_x86_64 dist/qiskit_aer*.whl
```
You can now go ahead and install the wheel stored in `wheelhouse/`.

Should you encounter a runtime error like
```
    Inconsistency detected by ld.so: dl-version.c: 205: _dl_check_map_versions: Assertion `needed != NULL' failed!
```
this means that your [patchelf](https://github.com/NixOS/patchelf) version (which is used by
`auditwheel` under the hood) is too old (https://github.com/pypa/auditwheel/issues/103)
Version `0.9` of `patchelf` is the earliest to include the patch
https://github.com/NixOS/patchelf/pull/85 which resolves this issue.
In the unlikely event that the `patchelf` package provided by your operating
system only provides an older version, fear not, because it is really easy to
[compile `patchelf` from source](https://github.com/NixOS/patchelf#compiling-and-testing).

Hopefully, this information was helpful.
In case you need more detailed information on some of the errors which may occur be sure to read
through https://github.com/Qiskit/qiskit-aer/issues/1033.

[1]: https://www.python.org/dev/peps/pep-0513/
[2]: https://www.python.org/dev/peps/pep-0571/
[3]: https://www.python.org/dev/peps/pep-0599/




## Useful CMake flags


There are some useful flags that can be set during CMake command invocation and
will help you change some default behavior. To make use of them, you just need to
pass them right after ``-D`` CMake argument. Example:

```
qiskit-aer/out$ cmake -DUSEFUL_FLAG=Value ..
```

In the case of building the Qiskit python extension, you have to pass these flags after writing
``--`` at the end of the python command line, eg:

```
qiskit-aer$ python ./setup.py bdist_wheel -- -DUSEFUL_FLAG=Value
```

These are the flags:

* USER_LIB_PATH

    This flag tells CMake to look for libraries that are needed by some of the native components to be built, but they are not in a common place where CMake could find it automatically.

    Values: An absolute path.
    Default: No value.
    Example: ``python ./setup.py bdist_wheel -- -DUSER_LIB_PATH=C:\path\to\openblas\libopenblas.so``

* AER_BLAS_LIB_PATH

    Tells CMake the directory to look for the BLAS library instead of the usual paths.
    If no BLAS library is found under that directory, CMake will raise an error and stop.

    It can also be set as an ENV variable with the same name, although the flag takes precedence.

    Values: An absolute path.
    Default: No value.
    Example: ``python ./setup.py bdist_wheel -- -DAER_BLAS_LIB_PATH=/path/to/look/for/blas/``

* BUILD_TESTS

    It will tell the build system to build C++ tests along with the simulator.

    Values: True|False
    Default: False
    Example: ``python ./setup.py bdist_wheel -- -DBUILD_TESTS=True``

* CMAKE_CXX_COMPILER

    This is an internal CMake flag. It forces CMake to use the provided toolchain to build everything.
    If it's not set, CMake system will use one of the toolchains installed in system.

    Values: g++|clang++|g++-8
    Default: Depends on the running platform and the toolchains installed
    Example: ``python ./setup.py bdist_wheel -- -DCMAKE_CXX_COMPILER=g++``

* AER_THRUST_BACKEND

    We use Thrust library for GPU support through CUDA. If we want to build a version of `Aer` with GPU acceleration, we need to install CUDA and set this variable to the value: "CUDA".
    There are other values that will use different CPU methods depending on the kind of backend we want to use:
    - "OMP": For OpenMP support
    - "TBB": For Intel Threading Building Blocks

    Values: CUDA|OMP|TBB
    Default: No value
    Example: ``python ./setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA``

* AER_CUDA_ARCH

    This flag allows us we to specify the CUDA architecture instead of letting the build system auto detect it.
    It can also be set as an ENV variable with the same name, although the flag takes precedence.

    Values:  Auto | Common | All | List of valid CUDA architecture(s).
    Default: Auto
    Example: ``python ./setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA -DAER_CUDA_ARCH="5.2; 5.3"``

* DISABLE_CONAN

    This flag allows disabling the Conan package manager. This will force CMake to look for
    the libraries in use on your system path, relying on FindPackage CMake mechanism and
    the appropriate configuration of libraries in order to use it.
    If a specific version is not found, the build system will look for any version available,
    although this may produce build errors or incorrect behaviour.

    __WARNING__: This is not the official procedure to build AER. Thus, the user is responsible
    of providing all needed libraries and corresponding files to make them findable to CMake.

    This is also available as the environment variable ``DISABLE_CONAN``, which overrides
    the CMake flag of the same name.

    Values: ON | OFF
    Default: OFF
    Example: ``python ./setup.py bdist_wheel -- -DDISABLE_CONAN=ON``

## Tests

Code contribution are expected to include tests that provide coverage for the
changes being made.

We have two types of tests in the codebase: Qiskit Terra integration tests and
Standalone integration tests.

For Qiskit Terra integration tests, you first need to build and install the Qiskit python extension, and then run `unittest` Python framework.

```
qiskit-aer$ pip install .
qiskit-aer$ stestr run
```

Manual for `stestr` can be found [here](https://stestr.readthedocs.io/en/latest/MANUAL.html#).

The integration tests for Qiskit python extension are included in: `test/terra`.

## C++ Tests

Our C++ unit tests use the Catch2 framework, an include-only C++ unit-testing framework. 
Catch2 framework documentation can be found [here](https://github.com/catchorg/Catch2).
Then, in any case, build Aer with the extra cmake argument BUILD_TESTS set to true:

```
python ./setup.py bdist_wheel --build-type=Debug -- -DBUILD_TESTS=True -- -j4 2>&1 |tee build.log
```

The test executable will be placed into the source test directory and can be run by:

```
qiskit-aer$ ./test/unitc_tests [Catch2-options]
```

## Platform support

Bear in mind that every new feature/change needs to be compatible with all our
supported platforms: Win64, MacOS (API Level >= 19) and Linux-x86_64. The
Continuous Integration (CI) systems will run builds and pass all the
corresponding tests to verify this compatibility.


## Debug

We have to build in debug mode if we want to start a debugging session with tools like `gdb` or `lldb`.
In order to create a Debug build for all platforms, we just need to pass a parameter while invoking the build to
create the wheel file:

    qiskit-aer$> python ./setup.py bdist_wheel --build-type=Debug

If you want to debug the standalone executable, then the parameter changes to:

    qiskit-aer/out$> cmake -DCMAKE_BUILD_TYPE=Debug

There are three different build configurations: `Release`, `Debug`, and `Release with Debug Symbols`, which parameters are:
`Release`, `Debug`, `RelWithDebInfo` respectively.

We recommend building in verbose mode and dump all the output to a file so it's easier to inspect possible build issues:

On Linux and Mac:

    qiskit-aer$ VERBOSE=1 python ./setup.py bdist_wheel --build-type=Debug 2>&1|tee build.log

On Windows:

    qisikt-aer> set VERBOSE=1
    qiskit-aer> python ./setup.py bdist_wheel --build-type=Debug 1> build.log 2>&1

We encourage to always send the whole `build.log` file when reporting a build issue, otherwise we will ask for it :)


**Stepping through the code**

Standalone version doesn't require anything special, just use your debugger like always:

    qiskit-aer/out/Debug$ gdb qasm_simulator

Stepping through the code of a Python extension is another story, trickier, but possible. This is because Python interpreters
usually load Python extensions dynamically, so we need to start debugging the python interpreter and set our breakpoints ahead of time, before any of our python extension symbols are loaded into the process.

Once built and installed we have to run the debugger with the python interpreter:

    $ lldb python

That will get us into the debugger (lldb in our case) interactive shell:

    (lldb) target create "python"
    Current executable set to 'python' (x86_64).
    (lldb)

Then we have to set our breakpoints:

    (lldb) b AER::controller_execute
    Breakpoint 1: no locations (pending).
    WARNING:  Unable to resolve breakpoint to any actual locations.

Here the message is clear, it can't find the function: `AER::controller_execute` because our python extension hasn't been loaded yet
 by the python interpreter, so it's "on-hold" hoping to find the function later in the execution.
Now we can run the python interpreter and pass the arguments (the python file to execute):

    (lldb) r test_qiskit_program.py
    Process 24896 launched: '/opt/anaconda3/envs/aer37/bin/python' (x86_64)
    3 locations added to breakpoint 1
    Executing on QasmSimulator for nq=16
    Process 24896 stopped
    * thread #12, stop reason = breakpoint 1.1
         frame #0: 0x000000012f834c10 controller_wrappers.cpython-37m-darwin.so`AER::Result AER::controller_execute<AER::Simulator::QasmController>(qobj_js=0x00007000032716b0) at controller_execute.hpp:48:16
         45
         46  	template <class controller_t>
         47  	Result controller_execute(const json_t &qobj_js) {
     ->  48  	  controller_t controller;
         49
         50  	  // Fix for MacOS and OpenMP library double initialization crash.
         51  	  // Issue: https://github.com/Qiskit/qiskit-aer/issues/1
    Target 0: (python) stopped.

After this, you can step through the code and continue with your debug session as always.
