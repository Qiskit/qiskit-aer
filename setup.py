# pylint: disable=invalid-name

"""
Main setup file for qiskit-aer
"""

import os
import subprocess
import sys

try:
    from skbuild import setup
except ImportError:
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'scikit-build'])
    from skbuild import setup

from setuptools import find_packages

requirements = [
    'numpy>=1.13',
    'scipy>=1.0',
    'cython>=0.27.1',
    'pybind11>=2.4'  # This isn't really an install requirement,
                     # Pybind11 is required to be pre-installed for
                     # CMake to successfully find header files.
                     # This should be fixed in the CMake build files.
]

setup_requirements = requirements + [
    'scikit-build',
    'cmake'
]

VERSION_PATH = os.path.join(os.path.dirname(__file__),
                            "qiskit", "providers", "aer", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()


def find_qiskit_aer_packages():
    """Finds qiskit aer packages.
    """
    location = 'qiskit/providers'
    prefix = 'qiskit.providers'
    aer_packages = find_packages(where=location)
    pkg_list = list(
        map(lambda package_name: '{}.{}'.format(prefix, package_name),
            aer_packages)
    )
    return pkg_list


setup(
    name='qiskit-aer',
    version=VERSION,
    packages=find_qiskit_aer_packages(),
    cmake_source_dir='.',
    description="Qiskit Aer - High performance simulators for Qiskit",
    url="https://github.com/Qiskit/qiskit-aer",
    author="AER Development Team",
    author_email="qiskit@us.ibm.com",
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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=requirements,
    setup_requires=setup_requirements,
    include_package_data=True,
    cmake_args=["-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.9"],
    keywords="qiskit aer simulator quantum addon backend",
    zip_safe=False
)
