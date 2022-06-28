# pylint: disable=invalid-name

"""
Main setup file for qiskit-aer
"""
import importlib
import inspect
import os
import setuptools
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
_DISABLE_DEPENDENCY_INSTALL = strtobool(os.getenv("DISABLE_DEPENDENCY_INSTALL", "OFF"))



def install_needed_req(import_name, package_name=None, min_version=None, max_version=None):
    if package_name is None:
        package_name = import_name
    install_ver = package_name
    if min_version:
        install_ver += '>=' + min_version
    if max_version:
        install_ver += '<' + max_version

    try:
        mod = importlib.import_module(import_name)
        mod_ver = parse_version(mod.__version__)
        if ((min_version and mod_ver < parse_version(min_version))
                or (max_version and mod_ver >= parse_version(max_version))):
            raise RuntimeError(f'{package_name} {mod_ver} is installed '
                               f'but required version is {install_ver}.')

    except ImportError as err:
        if _DISABLE_DEPENDENCY_INSTALL:
            raise ImportError(str(err) +
                              f"\n{package_name} is a required dependency. "
                              f"Please provide it and repeat install")

        subprocess.call([sys.executable, '-m', 'pip', 'install', install_ver])

if not _DISABLE_CONAN:
    install_needed_req('conans', package_name='conan', min_version='1.31.2')

install_needed_req('skbuild', package_name='scikit-build', min_version='0.11.0')
install_needed_req('pybind11', min_version='2.6')

from skbuild import setup


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

requirements = common_requirements + ['qiskit-terra>=0.20.0', 'scipy>=1.0']

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


cmake_args = []
is_win_32_bit = (platform.system() == 'Windows' and platform.architecture()[0] == "32bit")
if is_win_32_bit:
    cmake_args.append("-DCMAKE_GENERATOR_PLATFORM=Win32")

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
