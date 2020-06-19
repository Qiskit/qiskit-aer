# pylint: disable=invalid-name

"""
Main setup file for qiskit-aer
"""

import os
import subprocess
import sys
import inspect

try:
    from conans import client
except ImportError:
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

import setuptools


if not hasattr(setuptools,
               'find_namespace_packages') or not inspect.ismethod(
                    setuptools.find_namespace_packages):
    print("Your setuptools version:'{}' does not support PEP 420 "
          "(find_namespace_packages). Upgrade it to version >='40.1.0' and "
          "repeat install.".format(setuptools.__version__))
    sys.exit(1)

setup(
    packages=setuptools.find_namespace_packages(include=['qiskit.*']),
    cmake_source_dir='.',
    cmake_args=["-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.9"],
)
