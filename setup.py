# pylint: disable=invalid-name

"""
Main setup file for qiskit-aer
"""
import os
import platform

import setuptools
from skbuild import setup

PACKAGE_NAME = os.getenv("QISKIT_AER_PACKAGE_NAME", "qiskit-aer")
CUDA_MAJOR = os.getenv("QISKIT_AER_CUDA_MAJOR", "12")

# Allow build without the CUDA requirements. This is useful in case one intends to use a CUDA that exists in the host system.
ADD_CUDA_REQUIREMENTS = (
    False
    if os.getenv("QISKIT_ADD_CUDA_REQUIREMENTS", "true").lower() in ["false", "off", "no"]
    else True
)

extras_requirements = {"dask": ["dask", "distributed"]}

requirements = [
    "qiskit>=0.45.0",
    "numpy>=1.16.3",
    "scipy>=1.0",
    "psutil>=5",
]

classifiers = [
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
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering",
]


# ROCm is expected to be available in the target system to enable CDNA GPUs, so no
# requirements to be loaded. Also, no ROCm related classifiers are in place that
# could be used here.
if ADD_CUDA_REQUIREMENTS and "gpu" in PACKAGE_NAME and "rocm" not in PACKAGE_NAME:
    if "11" in CUDA_MAJOR:
        requirements_cuda = [
            "nvidia-cuda-runtime-cu11>=11.8.89",
            "nvidia-cublas-cu11>=11.11.3.6",
            "nvidia-cusolver-cu11>=11.4.1.48",
            "nvidia-cusparse-cu11>=11.7.5.86",
            "cuquantum-cu11>=23.3.0",
        ]
        classifiers_cuda = [
            "Environment :: GPU :: NVIDIA CUDA :: 11",
        ]
    else:
        requirements_cuda = [
            "nvidia-cuda-runtime-cu12>=12.1.105",
            "nvidia-nvjitlink-cu12",
            "nvidia-cublas-cu12>=12.1.3.1",
            "nvidia-cusolver-cu12>=11.4.5.107",
            "nvidia-cusparse-cu12>=12.1.0.106",
            "cuquantum-cu12>=23.3.0",
        ]
        classifiers_cuda = [
            "Environment :: GPU :: NVIDIA CUDA :: 12",
        ]
    requirements.extend(requirements_cuda)
    classifiers.extend(classifiers_cuda)

VERSION_PATH = os.path.join(os.path.dirname(__file__), "qiskit_aer", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = readme_file.read()


cmake_args = []
is_win_32_bit = platform.system() == "Windows" and platform.architecture()[0] == "32bit"
if is_win_32_bit:
    cmake_args.append("-DCMAKE_GENERATOR_PLATFORM=Win32")


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    packages=setuptools.find_packages(exclude=["test*"]),
    cmake_source_dir=".",
    description="Aer - High performance simulators for Qiskit",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Qiskit/qiskit-aer",
    author="AER Development Team",
    author_email="hello@qiskit.org",
    license="Apache 2.0",
    classifiers=classifiers,
    python_requires=">=3.7",
    install_requires=requirements,
    include_package_data=False,
    package_data={"qiskit_aer": ["VERSION.txt"], "qiskit_aer.library": ["*.csv"]},
    extras_require=extras_requirements,
    cmake_args=cmake_args,
    keywords="qiskit, simulator, quantum computing, backend",
    zip_safe=False,
    entry_points={
        "qiskit.transpiler.translation": [
            "aer_backend_plugin = qiskit_aer.backends.plugin.aer_backend_plugin:AerBackendPlugin",
        ]
    },
)
