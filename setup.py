# pylint: disable=invalid-name

"""
Main setup file for qiskit-aer
"""

import os
import subprocess
import sys
import inspect

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

requirements = [
    'qiskit-terra>=0.12.0',
    'numpy>=1.16.3',
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

class text_colors:
    """Text colors for terminal
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def _conda_blas_dir():
    """Finds the BLAS used in the NumPy installed via conda.
    """
    print(text_colors.HEADER + "="*80 + text_colors.ENDC)
    print(text_colors.HEADER + "Looking for Conda NumPy BLAS" + text_colors.ENDC)
    config = np.__config__
    blas_info = config.blas_opt_info
    has_lib_key = 'libraries' in blas_info.keys()
    blas = None
    if has_lib_key:
        blas_lib_dir = blas_info['library_dirs'][0]
    if hasattr(config, 'mkl_info') or \
            (has_lib_key and any('mkl' in lib for lib in blas_info['libraries'])):
        blas = 'INTEL MKL'
        blas_lib_name = ('libmkl_rt',)
    elif hasattr(config, 'openblas_info') or \
            (has_lib_key and any('openblas' in lib for lib in blas_info['libraries'])):
        blas = 'OPENBLAS'
        blas_lib_name = 'libopenblas'
    else:
        blas = None

    blas_file = None
    if blas:
        print(text_colors.OKGREEN + "BLAS Found: " + blas + text_colors.ENDC)

        if 'conda' in blas_lib_dir:
            #Go up a dir due to issues on Windows
            base_dir = os.path.join(os.path.dirname(blas_lib_dir))

            for subdir, _, files in os.walk(base_dir):
                for file in files:
                    if file.startswith(blas_lib_name):
                        if 'pkgs' not in subdir:
                            blas_file = file
                            blas_lib_dir = subdir
                            break
            if blas_file:
                print(text_colors.OKBLUE + "BLAS dir: " + blas_lib_dir + text_colors.ENDC)
                print(text_colors.OKGREEN + "BLAS executible: " + blas_file + text_colors.ENDC)
            else:
                print(text_colors.WARNING + \
                      "BLAS executible not found. Continuing without... " + \
                      text_colors.ENDC)
        else:
            print(text_colors.WARNING + \
                  "NumPy build without BLAS executible.  Continuing without..." + \
                  text_colors.ENDC)
    else:
        print(text_colors.WARNING + \
                  "No BLAS found. Continuing without... " + \
                  text_colors.ENDC)

    print(text_colors.HEADER + "="*80 + text_colors.ENDC)

    if blas_file:
        return blas_lib_dir
    return None

# check if wanting to use Conda NumPy BLAS
if "--with-conda-blas" in sys.argv:
    sys.argv.remove("--with-conda-blas")
    try:
        import numpy as np
    except:
        raise ImportError('NumPy must be pre-installed.')
    else:
        blas_dir = _conda_blas_dir()
        if blas_dir:
            if '--' not in sys.argv:
                sys.argv.append('--')
            sys.argv.append('-DBLAS_LIB_PATH='+blas_dir)

setup(
    name='qiskit-aer',
    version=VERSION,
    packages=setuptools.find_namespace_packages(include=['qiskit.*']),
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
