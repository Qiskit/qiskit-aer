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
try:
    import pybind11
except ImportError:
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'pybind11>=2.4'])

from setuptools import find_packages

requirements = [
    "numpy>=1.13",
    "pybind11>=2.4"
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

EXT_MODULES = []
# OpenPulse setup
if "--with-openpulse" in sys.argv:
    sys.argv.remove("--with-openpulse")
    # pylint: disable=ungrouped-imports
    import numpy as np
    import distutils.sysconfig
    from setuptools import Extension
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext # pylint: disable=unused-import

    INCLUDE_DIRS = [np.get_include()]
    # Add Cython OP extensions here
    OP_EXTS = ['channel_value', 'measure', 'memory', 'utils']
    # Add qutip_lite extensions here
    Q_EXTS = ['spmatfuncs', 'sparse_utils', 'spmath', 'math', 'spconvert']

    # Extra link args
    link_flags = []
    # If on Win and Python version >= 3.5 and not in MSYS2 (i.e. Visual studio compile)
    if (sys.platform == 'win32' and int(str(sys.version_info[0])+str(sys.version_info[1])) >= 35
            and os.environ.get('MSYSTEM') is None):
        compiler_flags = []
    # Everything else
    else:
        compiler_flags = ['-O2', '-funroll-loops']
        if sys.platform == 'darwin':
            # These are needed for compiling on OSX 10.14+
            compiler_flags.append('-mmacosx-version-min=10.9')
            link_flags.append('-mmacosx-version-min=10.9')

    # Remove -Wstrict-prototypes from cflags
    CFG_VARS = distutils.sysconfig.get_config_vars()
    if "CFLAGS" in CFG_VARS:
        CFG_VARS["CFLAGS"] = CFG_VARS["CFLAGS"].replace("-Wstrict-prototypes", "")

    # Add Cython files from cy
    for ext in OP_EXTS:
        _mod = Extension("qiskit.providers.aer.openpulse.cy."+ext,
                         sources=['qiskit/providers/aer/openpulse/cy/'+ext+'.pyx'],
                         include_dirs=[np.get_include()],
                         extra_compile_args=compiler_flags,
                         extra_link_args=link_flags,
                         language='c++')
        EXT_MODULES.append(_mod)

    for ext in Q_EXTS:
        _mod = Extension('qiskit.providers.aer.openpulse.qutip_lite.cy.'+ext,
                         sources=['qiskit/providers/aer/openpulse/qutip_lite/cy/'+ext+'.pyx',
                                  'qiskit/providers/aer/openpulse/qutip_lite/cy/src/zspmv.cpp'],
                         include_dirs=[np.get_include()],
                         extra_compile_args=compiler_flags,
                         extra_link_args=link_flags,
                         language='c++')
        EXT_MODULES.append(_mod)

    # Cythonize
    EXT_MODULES = cythonize(EXT_MODULES)

setup(
    name='qiskit-aer',
    version=VERSION,
    packages=find_qiskit_aer_packages(),
    cmake_source_dir='.',
    ext_modules=EXT_MODULES,
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
    setup_requires=['scikit-build', 'cmake', 'Cython', 'pybind11>2.4'],
    include_package_data=True,
    cmake_args=["-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.9"],
    keywords="qiskit aer simulator quantum addon backend",
    zip_safe=False
)
