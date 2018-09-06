Contributing
============

**We appreciate all kinds of help, so thank you!**

Contributing to the project
---------------------------

You can contribute in many ways to this project.

Issue reporting
~~~~~~~~~~~~~~~

This is a good point to start, when you find a problem please add
it to the `issue tracker <https://github.com/Qiskit/qiskit-terra/issues>`_.
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

Code
----

This section include some tips that will help you to push source code.

Dependencies
~~~~~~~~~~~~

Our build system is based on CMake, so we need to have `CMake 3.5 or higher <https://cmake.org/>`_
installed. As we will deal with languages that build native binaries, we will
need to have installed any of the `supported CMake build tools <https://cmake.org/cmake/help/v3.5/manual/cmake-generators.7.html>`_.

Mac
On Mac we have various options depending on the compiler we wanted to use.
Using Apple's clang compiler, we need to install an extra library for supporting
OpenMP: libomp. CMake build system will warn you otherwise. To install it manually
you can type:

.. code::

    $ brew install libomp

We do recommend to install OpenBLAS:

.. code::

    $ brew install openblas

Linux (Ubuntu >= 16.04)
Most of the major distributions come with a BLAS and LAPACK library implementation,
and this is enough to build all the simulators, but we do recommend using OpenBLAS
here as well, so in order to install it you have to type:

.. code::

    $ sudo apt install libopenblas-dev


Building
~~~~~~~~

The preferred way CMake is meant to be used, is by setting up an "out of source" build.
So in order to build our native code, we have to follow these steps:

All platforms

.. code::

    qiskit-aer$ mkdir out
    qiskit-aer$ cd out
    qiskit-aer/out$ cmake ..
    qiskit-aer/out$ cmake --build . --config Release -- -j4

NOTE: Even though latest versions of Windows have some sort of classic ``bash`` support
for command line operations, hence to commands above should work on latests versions,
the way to create directories on Windows is slightly different:
different

.. code::

    C:\..\> mkdir out
    C:\..\> cd out
    
Useful CMake flags
------------------

There are some useful flags that can be set during cmake command invocation and
will help you change some default behavior. To make use of them, you just need to
pass them right after ``-D`` cmake argument. Example:
.. code::

    qiskit-aer/out$ cmake -DUSEFUL_FLAG=Value ..

Flags:

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
    Tells the build system to build the tests as part of the building process.
    Values: True|False
    Default: False
    Example: ``cmake -DBUILD_TESTS=False ..``

Test
~~~~

New features often imply changes in the existent tests or new ones are
needed. Once they're updated/added run this be sure they keep passing.
Before running the tests, we have to build them. They are built by default
within the normal building process, and there's also a specific target just in
case you don't want to rebuild everthing:

.. code::

    qiskit-aer/out$ cmake --build . --target build_tests

For executing the tests, we will use the ``ctest`` tool like:

.. code::

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
  - The API of the SDK could change without prior notice.

- ``stable``

  - This is our stable release branch.
  - It's always synchronized with the latest distributed package, as for now,
    the package you can download from pip.
  - The code in this branch is well tested and should be free of errors
    (unfortunately sometimes it's not).
  - This is a stable branch (as the name suggest), meaning that you can expect
    stable software ready for production environments.
  - All the tags from the release versions are created from this branch.

Release cycle
~~~~~~~~~~~~~

TODO: Review

From time to time, we will release brand new versions of the Qiskit SDK. These
are well-tested versions of the software.

When the time for a new release has come, we will:

1. Merge the ``master`` branch with the ``stable`` branch.
2. Create a new tag with the version number in the ``stable`` branch.
3. Crate and distribute the pip package.
4. Change the ``master`` version to the next release version.
5. Announce the new version to the world!

The ``stable`` branch should only receive changes in the form of bug fixes, so the
third version number (the maintenance number: [major].[minor].[maintenance])
will increase on every new change.

What version should I use: development or stable?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO: TBD

Documentation
-------------

TODO: TBD