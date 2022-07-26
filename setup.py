# pylint: disable=invalid-name

"""
Main setup file for qiskit-aer
"""
import importlib
import inspect
import os
import setuptools
from skbuild import setup
import subprocess
import sys
from pkg_resources import parse_version
import platform


def strtobool(val):
    val_lower = val.lower()
    if val_lower in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val_lower in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f'Value: "{val}" not recognizes as True or False')


PACKAGE_NAME = os.getenv('QISKIT_AER_PACKAGE_NAME', 'qiskit-aer')
_DISABLE_CONAN = strtobool(os.getenv("DISABLE_CONAN", "OFF"))

# These are requirements that are both runtime/install dependencies and
# also build time/setup requirements and will be added to both lists
# of requirements
common_requirements = [
    'numpy>=1.16.3',
]

setup_requirements = common_requirements + [
    'scikit-build>=0.11.0',
    'cmake!=3.17,!=3.17.0',
    'pybind11>=2.6',
]

extras_requirements = {
    "dask": ["dask", "distributed"]
}

if not _DISABLE_CONAN:
    setup_requirements.append('conan>=1.22.2')

requirements = common_requirements + ['qiskit-terra>=0.21.0', 'scipy>=1.0']

VERSION_PATH = os.path.join(os.path.dirname(__file__),
                            "qiskit_aer", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()


cmake_args = []
is_win_32_bit = (platform.system() == 'Windows' and platform.architecture()[0] == "32bit")
if is_win_32_bit:
    cmake_args.append("-DCMAKE_GENERATOR_PLATFORM=Win32")

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    packages=setuptools.find_packages(include=['qiskit_aer.*']),
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    setup_requires=setup_requirements,
    include_package_data=True,
    extras_require=extras_requirements,
    cmake_args=cmake_args,
    keywords="qiskit aer simulator quantum addon backend",
    zip_safe=False
)
