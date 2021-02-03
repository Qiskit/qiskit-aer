# pylint: disable=invalid-name

"""
Main setup file for qiskit-aer
"""
import distutils.util
import importlib
import inspect
import os
import setuptools
import subprocess
import sys


PACKAGE_NAME = os.getenv('QISKIT_AER_PACKAGE_NAME', 'qiskit-aer')
_DISABLE_CONAN = distutils.util.strtobool(os.getenv("DISABLE_CONAN", "OFF").lower())

try:
    from Cython.Build import cythonize
except ImportError:
    import subprocess
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'Cython>=0.27.1'])
    from Cython.Build import cythonize

if not _DISABLE_CONAN:
    try:
        from conans import client
    except ImportError:
        # Problem with Conan and urllib3 1.26
        subprocess.call([sys.executable, '-m', 'pip', 'install', 'urllib3<1.26'])

        subprocess.call([sys.executable, '-m', 'pip', 'install', 'conan'])
        from conans import client

try:
    from skbuild import setup
except ImportError:
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'scikit-build'])
    from skbuild import setup

try:
    import pybind11
except ImportError:
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'pybind11>=2.4'])

try:
    from numpy import array
except ImportError:
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'numpy>=1.16.3'])

from skbuild import setup


# These are requirements that are both runtime/install dependencies and
# also build time/setup requirements and will be added to both lists
# of requirements
common_requirements = [
    'numpy>=1.16.3',
    'scipy>=1.0',
    'cython>=0.27.1',
    'pybind11>=2.4'  # This isn't really an install requirement,
                     # Pybind11 is required to be pre-installed for
                     # CMake to successfully find header files.
                     # This should be fixed in the CMake build files.
]

setup_requirements = common_requirements + [
    'scikit-build',
    'cmake!=3.17,!=3.17.0',
]
if not _DISABLE_CONAN:
    setup_requirements.append('urllib3<1.26')
    setup_requirements.append('conan>=1.22.2')

requirements = common_requirements + ['qiskit-terra>=0.12.0']

if not hasattr(setuptools,
               'find_namespace_packages') or not inspect.ismethod(
                    setuptools.find_namespace_packages):
    print("Your setuptools version:'{}' does not support PEP 420 "
          "(find_namespace_packages). Upgrade it to version >='40.1.0' and "
          "repeat install.".format(setuptools.__version__))
    sys.exit(1)

VERSION_PATH = os.path.join(os.path.dirname(__file__),
                            "qiskit", "providers", "aer", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    packages=setuptools.find_namespace_packages(include=['qiskit.*']),
    cmake_source_dir='.',
    description="Qiskit Aer - High performance simulators for Qiskit",
    long_description=README,
    long_description_content_type='text/markdown',
    url="https://github.com/Qiskit/qiskit-aer",
    author="AER Development Team",
    author_email="hello@qiskit.org",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    setup_requires=setup_requirements,
    include_package_data=True,
    cmake_args=["-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.9"],
    keywords="qiskit aer simulator quantum addon backend",
    zip_safe=False
)
