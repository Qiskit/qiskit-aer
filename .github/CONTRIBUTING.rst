Contributing
============

**We appreciate all kinds of help, so thank you!**

Contributing to the project
---------------------------

You can contribute in many ways to this project.

Issue reporting
~~~~~~~~~~~~~~~

This is a good point to start, when you find a problem please add
it to the `issue tracker <https://github.com/Qiskit/qiskit-aer/issues>`_.
The ideal report should include the steps to reproduce it.

Doubts solving
~~~~~~~~~~~~~~

To help less advanced users is another wonderful way to start. You can
help us close some opened issues. This kind of tickets should be
labeled as ``question``.

Improvement proposal
~~~~~~~~~~~~~~~~~~~~

If you have an idea for a new feature please open a ticket labeled as
``enhancement``. If you could also add a piece of code with the idea
or a partial implementation it would be awesome.

Contributor License Agreement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We'd love to accept your code! Before we can, we have to get a few legal
requirements sorted out. By signing a contributor license agreement (CLA), we
ensure that the community is free to use your contributions.

When you contribute to the Qiskit project with a new pull request, a bot will
evaluate whether you have signed the CLA. If required, the bot will comment on
the pull request,  including a link to accept the agreement. The
`individual CLA <https://qiskit.org/license/qiskit-cla.pdf>`_ document is
available for review as a PDF.

NOTE: If you work for a company that wants to allow you to contribute your work,
then you'll need to sign a `corporate CLA <https://qiskit.org/license/qiskit-corporate-cla.pdf>`_
and email it to us at qiskit@us.ibm.com.


Pre-requisites
~~~~~~~~~~~~~~

Most of the required dependencies can be installed via ``pip``, using the
``requirements-dev.txt`` file, eg:

.. code:: sh

    pip install -U -r requirements-dev.txt

As we are dealing with languages that build to native binaries, we will
need to have installed any of the `supported CMake build tools <https://cmake.org/cmake/help/v3.5/manual/cmake-generators.7.html>`_.

We do support most of the common available toolchains like: gcc, clang, Visual Studio.
The only required requisite is that the toolchain needs to support C++14.

**Mac**

On Mac we have various options depending on the compiler we want to use.
If we want to use Apple's Clang compiler, we need to install an extra library for
supporting OpenMP: libomp. The CMake build system will warn you otherwise.
To install it manually:
you can type:

.. code::

    $ brew install libomp

We do recommend installing OpenBLAS, which is our default choice:

.. code::

    $ brew install openblas

CMake build system will search for other BLAS implementation alternatives if
OpenBLAS is not installed in the system.

You further need to have Command Line Tools installed on MacOS:

.. code::
   
   $ xcode-select --install


**Linux (Ubuntu >= 16.04)**

Most of the major distributions come with a BLAS and LAPACK library implementation,
and this is enough to build all the simulators, but we do recommend using OpenBLAS
here as well, so in order to install it you have to type:

.. code::

    $ sudo apt install libopenblas-dev

**Windows**

On Windows you must have Anaconda3 installed in the system, and We recommend installing
Visual Studio 2017 (Communit Edition).
The same rules applies when searching for an OpenBLAS implementation, if CMake can't
find one suitable implementation installed in the system, it will take the BLAS
library from the Anaconda3 environment.




Building
~~~~~~~~

There are two ways of building Aer simulators, depending on our goal they are:
1. Build Terra compatible addon.
2. Build standalone executable

**Terra addon**

For the former, we just need to call the ``setup.py`` script:

.. code::

  qiskit-aer$ python ./setup.py bdist_wheel


We are using `scikit-build <https://scikit-build.readthedocs.io/en/latest/>`_ as a substitute of `setuptools`.
This is basically the glue between ``setuptools`` and ``CMake``, so there are various options to pass variables to ``CMake``, and 
the undelying build system (depending on your platform). The way to pass variables is:

.. code::

    qiskit-aer$ python ./setup.py bdist_wheel -- -DCMAKE_VARIABLE=Values -- -Makefile_or_VisuaStudio_Flag
    
So a real example could be:

.. code::

    qiskit-aer$ python ./setup.py bdist_wheel -- -j8
    
This is setting the CMake variable ``STATIC_LINKING`` to value ``True`` so CMake will try to create an statically linked cython
library, and is passing ``-j8`` flag to the underlaying build system, which in this case is Makefile, telling it that we want to
build in parallel, using 8 processes.

*N.B. on MacOS:*, you may need to turn off static linking and specify your platform name, e.g.:

.. code::

   qiskit-aer$ python ./setup.py bdist_wheel --plat-name macosx-10.9-x86_64 -- -DSTATIC_LINKING=False -- -j8


After this command is executed successfully, we will have a wheel package into the ``dist/`` directory, so next step is installing it:

.. code::

  qiskit-aer/$ cd dist
  qiskit-aer/dist$ pip install qiskit_aer-<...>.whl


**Standalone executable**

If we want to build an standalone executable, we have to use CMake directly.
The preferred way CMake is meant to be used, is by setting up an "out of source" build.
So in order to build our standalone executable, we have to follow these steps:

All platforms

.. code::

    qiskit-aer$ mkdir out
    qiskit-aer$ cd out
    qiskit-aer/out$ cmake ..
    qiskit-aer/out$ cmake --build . --config Release -- -j4

Once built, you will have your standalone executable into the ``Release`` or ``Debug``
directory (depending on the type of building choosen with the ``--config`` option):

.. code::

  qiskit-aer/out$ cd Release
  qiskit-aer/out/Release$ ls
  aer_simulator_cpp



Useful CMake flags
------------------

There are some useful flags that can be set during cmake command invocation and
will help you change some default behavior. To make use of them, you just need to
pass them right after ``-D`` cmake argument. Example:

.. code::

    qiskit-aer/out$ cmake -DUSEFUL_FLAG=Value ..

In the case of building the Terra addon, you have to pass these flags after writing
``--`` at the end of the python command line, eg:

.. code::

  qiskit-aer$ python ./setup.py bdist_wheel -- -DUSEFUL_FLAG=Value


These are the flags:

USER_LIB_PATH
    This flag tells CMake to look for libraries that are needed by some of the native
    components to be built, but they are not in a common place where CMake could find
    it automatically.

    Values: An absolute path with file included.
    Default: No value.
    Example: ``cmake -DUSER_LIB_PATH=C:\path\to\openblas\libopenblas.so ..``

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


Tests
~~~~~

Almost every code contribution should be accompained by it's corresponding set of tests.
You won't probably hear complaints if there are too many tests in your PR :), but the other
way around is unnacceptable :(
We have two types of tests in the codebase: Qiskit Terra integration tests and Standalone integration tests.

For Qiskit Terra integration tests, you first need to build and install the Terra addon,
and then run `unittest` Python framework.

.. code::

  qiskit-aer$ python ./setup.py install
  # if you had to use --plat-name macosx-10.9-x86_64 for bdist_wheel then you need to do this for install:
  #   python ./setup.py install -- -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.9 -DCMAKE_OSX_ARCHITECTURES:STRING=x86_64
  qiskit-aer$ python -m unittest discover -s test -v

The integration tests for Terra addon are included in: `test/terra`.


For the Standalone version of the simulator, we have C++ tests that use the Catch library.
Tests are located in `test/src` directory, and in order to run them, you have to build them first:

.. code::

  qiskit-aer$ mkdir out
  qiskit-aer$ cd out
  qiskit-aer/out$ cmake .. -DBUILD_TESTS=True
  qiskit-aer/out$ cmake --build . --config Release -- -j4
  qiskit-aer/out$ ctest -VV


Style guide
~~~~~~~~~~~

Please submit clean code and please make effort to follow existing conventions
in order to keep it as readable as possible.

TODO: Decide code convention

A linter (clang-tidy) is passed automatically every time a building is
invoqued. It will stop the current build if detects style erros, or common pitfalls.


Good first contributions
~~~~~~~~~~~~~~~~~~~~~~~~

You are welcome to contribute wherever in the code you want to, of course, but
we recommend taking a look at the "Good first contribution" label into the
issues and pick one. We would love to mentor you!

Doc
~~~

Review the parts of the documentation regarding the new changes and update it
if it's needed.

Pull requests
~~~~~~~~~~~~~

We use `GitHub pull requests <https://help.github.com/articles/about-pull-requests>`_
to accept the contributions.

A friendly reminder! We'd love to have a previous discussion about the best way to
implement the feature/bug you are contributing with. This is a good way to
improve code quality in our beloved simulators!, so remember to file a new Issue before
starting to code for a solution.

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

Pull request checklist
""""""""""""""""""""""

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

Commit messages
"""""""""""""""

Please follow the next rules for the commit messages:

- It should include a reference to the issue ID in the first line of the commit,
  **and** a brief description of the issue, so everybody knows what this ID
  actually refers to without wasting to much time on following the link to the
  issue.

- It should provide enough information for a reviewer to understand the changes
  and their relation to the rest of the code.

A good example:

.. code::

    Issue #190: Short summary of the issue
    * One of the important changes
    * Another important change

A (really) bad example:

.. code::

    Fixes #190

Development cycle
-----------------

TODO: Review

Our development cycle is straightforward, we define a roadmap with milestones
for releases, and features that we want to include in these releases. The
roadmap is not public at the moment, but it's a committed project in our
community and we are working to make parts of it public in a way that can be
beneficial for everyone. Whenever a new release is close to be launched, we'll
announce it and detail what has changed since the latest version.
The channels we'll use to announce new releases are still being discussed, but
for now you can `follow us <https://twitter.com/qiskit>`_ on Twitter!

Branch model
~~~~~~~~~~~~

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

