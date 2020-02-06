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
    $ source activate QiskitDevEnv

Clone the Qiskit Aer repo via *git*.

    $ git clone https://github.com/Qiskit/qiskit-aer

Most of the required dependencies can be installed via *pip*, using the
`requirements-dev.txt` file, e.g.:

    $ cd qiskit-aer
    $ pip install -r requirements-dev.txt


### Linux

Qiskit is supported on Ubuntu >= 16.04. To get most of the necessary compilers
and libraries, install the *build-essential* package by running

    $ sudo apt install build-essential

Although the *BLAS* and *LAPACK* library implementations included in the
*build-essential* package are sufficient to build all of the Aer simulators, we
recommend using *OpenBLAS*, which you can install by running

    $ sudo apt install libopenblas-dev

There are two ways of building Aer simulators, depending on your goal:

1. Build a Terra compatible add-on;
2. Build a standalone executable.

**Terra Add-on**

For the former, we just need to call the `setup.py` script:

    qiskit-aer$ python ./setup.py bdist_wheel

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

    qiskit-aer$ python ./setup.py bdist_wheel -- -- -j8

This is passing the flag `-j8` to the underlying build system, which in this
case is *Makefile*, telling it that we want to build in parallel using 8
processes.

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

There are two ways of building Aer simulators, depending on your goal:

1. Build a Terra compatible add-on;
2. Build a standalone executable.

**Terra Add-on**

For the former, we just need to call the `setup.py` script:

    qiskit-aer$ python ./setup.py bdist_wheel

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

    qiskit-aer$ python ./setup.py bdist_wheel -- -- -j8

This is passing the flag `-j8` to the underlying build system, which in this
case is *Makefile*, telling it that we want to build in parallel using 8
processes.

> You may need to specify your platform name and turn off static linking, for
> example:

    qiskit-aer$ python ./setup.py bdist_wheel --plat-name macosx-10.9-x86_64 \
    -- -DSTATIC_LINKING=False -- -j8

Here `--plat-name` is a flag to *setuptools*, `-DSTATIC_LINKING` is a flag to
*CMake*, and `-j8` is a flag to the underlying build system.

After this command is executed successfully, we will have a wheel package into
the `dist/` directory, so next step is installing it:

    qiskit-aer/$ cd dist
    qiskit-aer/dist$ pip install qiskit_aer-<...>.whl

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



### Windows

On Windows, you must have *Anaconda3* installed. We recommend also installing
*Visual Studio 2017* (Community Edition). *Anaconda3* is required when
searching for an *OpenBLAS* implementation. If *CMake* can't find a suitable
implementation installed, it will take the *BLAS* library from the
*Anaconda3* environment.



## Useful CMake flags


There are some useful flags that can be set during cmake command invocation and
will help you change some default behavior. To make use of them, you just need to
pass them right after ``-D`` cmake argument. Example:

```
qiskit-aer/out$ cmake -DUSEFUL_FLAG=Value ..
```

In the case of building the Terra addon, you have to pass these flags after writing
``--`` at the end of the python command line, eg:

```
qiskit-aer$ python ./setup.py bdist_wheel -- -DUSEFUL_FLAG=Value
```

These are the flags:

USER_LIB_PATH
    This flag tells CMake to look for libraries that are needed by some of the native
    components to be built, but they are not in a common place where CMake could find
    it automatically.

    Values: An absolute path with file included.
    Default: No value.
    Example: ``cmake -DUSER_LIB_PATH=C:\path\to\openblas\libopenblas.so ..``

BLAS_LIB_PATH
    Tells CMake the directory to look for the BLAS library instead of the usual paths.
    If no BLAS library is found under that directory, CMake will raise an error and stop.

    Values: An absolute path with file included.
    Default: No value.
    Example: ``cmake -DBLAS_LIB_PATH=/path/to/look/for/blas/ ..``

STATIC_LINKING
    Tells the build system whether to create static versions of the programs being built or not.
    NOTE: On MacOS static linking is not fully working for all versions of GNU G++/Clang
    compilers, so depending on the version of the compiler installed in the system,
    enable this flag in this platform could cause errors.

    Values: True|False
    Default: False
    Example: ``cmake -DSTATIC_LINKING=True ..``

BUILD_TESTS
    It will tell the build system to build C++ tests along with the simulator.

    Values: True|False
    Default: False
    Example: ``cmake -DBUILD_TESTS=True ..``

CMAKE_CXX_COMPILER
    This is an internal CMake flag. It forces CMake to use the provided toolchain to build everthing.
    If it's not set, CMake system will use one of the toolchains installed in system.

    Values: g++|clang++|g++-8
    Default: Depends on the running platform and the toolchains installed
    Example: ``cmake -DCMAKE_CXX_COMPILER=g++``

AER_THRUST_BACKEND
    We use Thrust library for GPU support through CUDA. If we want to build a version of Aer with GPU acceleration, we need to install CUDA and set this variable to the value: "CUDA".
    There are other values that will use different CPU methods depending on the kind of backend we want to use:
    - "OMP": For OpenMP support
    - "TBB": For Intel Threading Building Blocks

    Values: CUDA|OMP|TTB
    Default: No value
    Example: ``cmake -DAER_THRUST_BACKEND=CUDA``




## Tests

Almost every code contribution should be accompained by it's corresponding set of tests.
You won't probably hear complaints if there are too many tests in your PR :), but the other
way around is unnacceptable :(
We have two types of tests in the codebase: Qiskit Terra integration tests and Standalone integration tests.

For Qiskit Terra integration tests, you first need to build and install the Terra addon,
and then run `unittest` Python framework.

```
qiskit-aer$ python ./setup.py install
# if you had to use --plat-name macosx-10.9-x86_64 for bdist_wheel then you need to do this for install:
#   python ./setup.py install -- -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.9 -DCMAKE_OSX_ARCHITECTURES:STRING=x86_64
qiskit-aer$ python -m unittest discover -s test -v
```

Alternatively you can run the integration tests in parallel using `stestr`.
```
qiskit-aer$ stestr run --slowest
```

The `slowest` option will print the slowest tests at the end. 
Manual for `stestr` can be found [here](https://stestr.readthedocs.io/en/latest/MANUAL.html#).
You may need to install it:
```
qiskit-aer$ pip install stestr
```

The integration tests for Terra addon are included in: `test/terra`.


For the Standalone version of the simulator, we have C++ tests that use the Catch library.
Tests are located in `test/src` directory, and in order to run them, you have to build them first:

```
qiskit-aer$ mkdir out
qiskit-aer$ cd out
qiskit-aer/out$ cmake .. -DBUILD_TESTS=True
qiskit-aer/out$ cmake --build . --config Release -- -j4
qiskit-aer/out$ ctest -VV
```

## Style guide


Please submit clean code and please make effort to follow existing conventions
in order to keep it as readable as possible.

TODO: Decide code convention

A linter (clang-tidy) is passed automatically every time a building is
invoqued. It will stop the current build if detects style erros, or common pitfalls.


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
  execute ``make style`` and ``make lint`` locally, which will print potential
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

- ``stable``

  - This is our stable release branch.
  - It's always synchronized with the latest distributed package, as for now,
    the package you can download from pip.
  - The code in this branch is well tested and should be free of errors
    (unfortunately sometimes it's not).
  - This is a stable branch (as the name suggest), meaning that you can expect
    stable software ready for production environments.
  - All the tags from the release versions are created from this branch.

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
