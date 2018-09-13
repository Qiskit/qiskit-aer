from skbuild import setup
from setuptools import find_packages

setup(
    name='qiskit_addon_qv', # This will change once Terra accepts addons
    packages=['qiskit_addon_qv'],
    cmake_source_dir='../..',
    description="AER - QV Quantum Simulator addon for Qiskit",
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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit simulator quantum addon backend"
)
