# Contributing

We appreciate all kinds of help, so thank you!

## Issue Reporting

This is a good point to start, when you find a problem please add
it to the [issue tracker](https://github.com/Qiskit/qiskit-aer/issues).
The ideal report should include the steps to reproduce it.

## Doubts Solving

To help less advanced users is another wonderful way to start. You can
help us close some opened issues. This kind of tickets should be
labeled as `question`.

## Improvement Proposal

If you have an idea for a new feature please open a ticket labeled as
`enhancement`. If you could also add a piece of code with the idea
or a partial implementation it would be awesome.

## Contributor License Agreement


We'd love to accept your code! Before we can, we have to get a few legal
requirements sorted out. By signing a contributor license agreement (CLA), we
ensure that the community is free to use your contributions.

When you contribute to the Qiskit project with a new pull request, a bot will
evaluate whether you have signed the CLA. If required, the bot will comment on
the pull request,  including a link to accept the agreement. The
[individual CLA](https://qiskit.org/license/qiskit-cla.pdf) document is
available for review as a PDF.

> Note: If you work for a company that wants to allow you to contribute your
> work, then you'll need to sign a [corporate
> CLA](https://qiskit.org/license/qiskit-corporate-cla.pdf) and email it to us
> at qiskit@us.ibm.com.


## Install from Source

>  Note: The following are prerequisites for all operating systems

We recommend using Python virtual environments to cleanly separate Qiskit from
other applications and improve your experience.

The simplest way to use environments is by using *Anaconda* in a terminal
window

    $ conda create -y -n QiskitDevEnv python=3
    $ conda activate QiskitDevEnv

Clone the `Qiskit Aer` repo via *git*.

    $ git clone https://github.com/Qiskit/qiskit-aer

Most of the required dependencies can be installed via *pip*, using the
`requirements-dev.txt` file, e.g.:

    $ cd qiskit-aer
    $ pip install -r requirements-dev.txt


### Linux

Qiskit is officially supported on Red Hat, CentOS, Fedora and Ubuntu distributions, as long as you can install a GCC version that is C++14 compatible and the few dependencies we need.

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
    aer_simulator_cpp


**Advanced options**

Because the standalone version of `Aer` doesn't need Python at all, the build system is
based on CMake, just like most of other C++ projects. So in order to pass all the different
options we have on `Aer` to CMake we use it's native mechanism:

    qiskit-aer/out$ cmake -DCMAKE_CXX_COMPILER=g++-9 -DBLAS_LIB_PATH=/path/to/my/blas ..


### macOS

There are various methods depending on the compiler we want to use. If we want
to use the *Clang* compiler, we need to install an extra library for
supporting *OpenMP*: *libomp*. The *CMake* build system will warn you
otherwise. To install it manually, in a terminal window, run:

    $ brew install libomp

We recommend installing *OpenBLAS*, which is our default choice:

    $ brew install openblas

The *CMake* build system will search for other *BLAS* implementation
alternatives if *OpenBLAS* is not installed in the system.

You further need to have *Xcode Command Line Tools* installed on macOS:

    $ xcode-select --install

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
    aer_simulator_cpp

***Advanced options***

Because the standalone version of `Aer` doesn't need Python at all, the build system is
based on CMake, just like most of other C++ projects. So in order to pass all the different
options we have on `Aer` to CMake we use it's native mechanism:

    qiskit-aer/out$ cmake -DCMAKE_CXX_COMPILER=g++-9 -DBLAS_LIB_PATH=/path/to/my/blas ..



### Windows

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
    aer_simulator_cpp

***Advanced options***

Because the standalone version of `Aer` doesn't need Python at all, the build system is
based on CMake, just like most of other C++ projects. So in order to pass all the different
options we have on `Aer` to CMake we use it's native mechanism:

    (QiskitDevEnv) qiskit-aer\out> cmake -G "Visual Studio 15 2017" -DBLAS_LIB_PATH=c:\path\to\my\blas ..



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

* BLAS_LIB_PATH

    Tells CMake the directory to look for the BLAS library instead of the usual paths.
    If no BLAS library is found under that directory, CMake will raise an error and stop.

    Values: An absolute path.
    Default: No value.
    Example: ``python ./setup.py bdist_wheel -- -DBLAS_LIB_PATH=/path/to/look/for/blas/``

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




## Tests

Almost every code contribution should be accompanied by it's corresponding set of tests.
You won't probably hear complaints if there are too many tests in your PR :), but the other
way around is unacceptable :(
We have two types of tests in the codebase: Qiskit Terra integration tests and Standalone integration tests.

For Qiskit Terra integration tests, you first need to build and install the Qiskit python extension,
and then run `unittest` Python framework.

```
qiskit-aer$ pip install .
qiskit-aer$ stestr run
```

Manual for `stestr` can be found [here](https://stestr.readthedocs.io/en/latest/MANUAL.html#).

The integration tests for Qiskit python extension are included in: `test/terra`.


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


## Style guide


Please submit clean code and please make effort to follow existing conventions
in order to keep it as readable as possible.

TODO: Decide code convention

A Python linter and C++ linter is passed automatically every time a Pull Request
or a commit is pushed to the repository. bIt will stop the current build if detects
style errors, or common pitfalls.
Bare in mind that every new feature/change needs to be compatible with all our
supported platforms: Win64, MacOS (API Level >= 19) and Linux-x86_64. The Continuous
Integration (CI) systems will run builds and pass all the corresponding tests to
verify this compatibility.


## Good First Contributions

You are welcome to contribute wherever in the code you want to, of course, but
we recommend taking a look at the "Good first contribution" label into the
issues and pick one. We would love to mentor you!

## Doc

Review the parts of the documentation regarding the new changes and update it
if it's needed.

## Pull Requests

We use [GitHub pull
requests](https://help.github.com/articles/about-pull-requests) to accept the
contributions.

A friendly reminder! We'd love to have a previous discussion about the best way
to implement the feature/bug you are contributing with. This is a good way to
improve code quality in our beloved simulators!, so remember to file a new Issue
before starting to code for a solution.

So after having discussed the best way to land your changes into the codebase,
you are ready to start coding (yay!). We have two options here:

1. You think your implementation doesn't introduce a lot of code, right?. Ok,
   no problem, you are all set to create the PR once you have finished coding.
   We are waiting for it!
2. Your implementation does introduce many things in the codebase. That sounds
   great! Thanks!. In this case you can start coding and create a PR with the
   word: **[WIP]** as a prefix of the description. This means "Work In
   Progress", and allow reviewers to make micro reviews from time to time
   without waiting to the big and final solution... otherwise, it would make
   reviewing and coming changes pretty difficult to accomplish. The reviewer
   will remove the **[WIP]** prefix from the description once the PR is ready
   to merge.

### Pull Request Checklist


When submitting a pull request and you feel it is ready for review, please
double check that:

* the code follows the code style of the project. For convenience, you can
  execute ``pycodestyle --ignore=E402,W504 --max-line-length=100 qiskit/providers/aer``
  and ``pylint -j 2 -rn qiskit/providers/aer`` locally, which will print potential
  style warnings and fixes.
* the documentation has been updated accordingly. In particular, if a function
  or class has been modified during the PR, please update the docstring
  accordingly.
* your contribution passes the existing tests, and if developing a new feature,
  that you have added new tests that cover those changes.
* you add a new line to the ``CHANGELOG.rst`` file, in the ``UNRELEASED``
  section, with the title of your pull request and its identifier (for example,
  "``Replace OldComponent with FluxCapacitor (#123)``".

### Commit Messages


Please follow the next rules for the commit messages:

- It should include a reference to the issue ID in the first line of the commit,
  **and** a brief description of the issue, so everybody knows what this ID
  actually refers to without wasting to much time on following the link to the
  issue.

- It should provide enough information for a reviewer to understand the changes
  and their relation to the rest of the code.

A good example:

```
Issue #190: Short summary of the issue
* One of the important changes
* Another important change
```

A (really) bad example:

```
Fixes #190
```

## Development Cycle


TODO: Review

Our development cycle is straightforward, we define a roadmap with milestones
for releases, and features that we want to include in these releases. The
roadmap is not public at the moment, but it's a committed project in our
community and we are working to make parts of it public in a way that can be
beneficial for everyone. Whenever a new release is close to be launched, we'll
announce it and detail what has changed since the latest version. The channels
we'll use to announce new releases are still being discussed, but for now you
can [follow us](https://twitter.com/qiskit) on Twitter!

## Branch Model


There are two main branches in the repository:

- ``master``

  - This is the development branch.
  - Next release is going to be developed here. For example, if the current
    latest release version is r1.0.3, the master branch version will point to
    r1.1.0 (or r2.0.0).
  - You should expect this branch to be updated very frequently.
  - Even though we are always doing our best to not push code that breaks
    things, is more likely to eventually push code that breaks something...
    we will fix it ASAP, promise :).
  - This should not be considered as a stable branch to use in production
    environments.
  - The public interface could change without prior notice.

- ``stable/<major>.<minor>``

  - This is our stable release branch.
  - It's always synchronized with the latest distributed package, as for now,
    the package you can download from pip.
  - The code in this branch is well tested and should be free of errors
    (unfortunately sometimes it's not).
  - This is a stable branch (as the name suggest), meaning that you can expect
    stable software.
  - Every time there's a new minor bump (https://semver.org/) we will branch off ``master``
    to create a new release branch, for example: ``stable/1.1``.

Stable Branch Policy
====================

The stable branch is intended to be a safe source of fixes for high impact bugs and security issues which have been fixed on master since a release. When reviewing a stable branch PR we need to balance the risk of any given patch with the value that it will provide to users of the stable branch. Only a limited class of changes are appropriate for inclusion on the stable branch. A large, risky patch for a major issue might make sense. As might a trivial fix for a fairly obscure error handling case. A number of factors must be weighed when considering a change:

- The risk of regression: even the tiniest changes carry some risk of breaking something and we really want to avoid regressions on the stable branch
- The user visible benefit: are we fixing something that users might actually notice and, if so, how important is it?
- How self-contained the fix is: if it fixes a significant issue but also refactors a lot of code, it’s probably worth thinking about what a less risky fix might look like
- Whether the fix is already on master: a change must be a backport of a change already merged onto master, unless the change simply does not make sense on master.

Backporting procedure:
----------------------

When backporting a patch from master to stable we want to keep a reference to the change on master. When you create the branch for the stable PR you can use:

`$ git cherry-pick -x $master_commit_id`

However, this only works for small self contained patches from master. If you
need to backport a subset of a larger commit (from a squashed PR for
example) from master this just need be done manually. This should be handled
by adding::

    Backported from: #master pr number

in these cases, so we can track the source of the change subset even if a
strict cherry pick doesn't make sense.

If the patch you’re proposing will not cherry-pick cleanly, you can help by resolving the conflicts yourself and proposing the resulting patch. Please keep Conflicts lines in the commit message to help review of the stable patch.

Backport Tags
-------------

Bugs or PRs tagged with `stable backport potential` are bugs which apply to the
stable release too and may be suitable for backporting once a fix lands in
master. Once the backport has been proposed, the tag should be removed.

The PR against the stable branch should include `[stable]` in the title, as a
sign that setting the target branch as stable was not a mistake. Also,
reference to the PR number in master that you are porting.


Troubleshooting
---------------

