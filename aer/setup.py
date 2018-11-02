import os
from skbuild import setup
from setuptools import find_packages

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "qiskit_aer")
with open(os.path.join(ROOT_DIR, "VERSION.txt"), "r") as version_file:
    VERSION = version_file.read().strip()

setup(
    name='qiskit_aer',
    version=VERSION,
    packages=find_packages(),
    cmake_source_dir='..',
    description="Qiskit Aer - High performance simulators for Qiskit",
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
    include_package_data=True,
    keywords="qiskit aer simulator quantum addon backend"
)
