# pylint: disable=invalid-name

"""
Main setup file for qiskit-aer
"""
import os
import pathlib
import platform

import subprocess
import setuptools
from skbuild import setup

DEBUG_MODE = os.environ.get("DEBUG") is not None
PACKAGE_NAME = os.getenv("QISKIT_AER_PACKAGE_NAME", "qiskit-aer")
CUDA_MAJOR = os.getenv("QISKIT_AER_CUDA_MAJOR", "12")

# Allow build without the CUDA requirements. This is useful in case one intends to use a CUDA that exists in the host system.
ADD_CUDA_REQUIREMENTS = (
    False
    if os.getenv("QISKIT_ADD_CUDA_REQUIREMENTS", "true").lower() in ["false", "off", "no"]
    else True
)

requirements = [
    "qiskit>=1.1.0",
    "numpy>=1.16.3",
    "scipy>=1.0",
    "psutil>=5",
    "python-dateutil>=2.8.0",
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
            "cuquantum-cu11>=23.3.0,<24.11.0",
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
            "cuquantum-cu12>=23.3.0,<24.11.0",
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

# run Conan
BUILD_DIR = os.path.join(os.path.dirname(__file__), "build")
os.makedirs(BUILD_DIR, exist_ok=True)
print("PATH:", os.environ.get("PATH"), flush=True)
print("CC:", os.environ.get("CC"), flush=True)
print("CXX:", os.environ.get("CXX"), flush=True)
try:
    result = subprocess.run(
        ["gcc", "-dumpmachine"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(f"Detected GCC machine triple: {result.stdout.strip()}", flush=True)
    result = subprocess.run(
        ["gcc", "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(f"gcc --version gives: {result.stdout.strip()}", flush=True)
    result = subprocess.run(
        [os.environ.get("CC"), "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(f"{os.environ.get("CC")} --version gives: {result.stdout.strip()}", flush=True)
    
except Exception as e:
    print(f"Failed to run 'gcc -dumpmachine':", flush=True)
try:
    subprocess.check_call(["conan", "profile", "detect", "--force"], cwd=BUILD_DIR)
    print("CONAN: New profile generated", flush=True)
except subprocess.CalledProcessError:
    print("CONAN: profile already exists", flush=True)

conan_profile_path = pathlib.Path.home() / ".conan2" / "profiles" / "default"
if conan_profile_path.exists():
    print(f"CONAN: Profile found:", flush=True)
    print(conan_profile_path.read_text(), flush=True)
else:
    print(f"CONAN: Profile not found", flush=True)

subprocess.check_call(
    [
        "conan",
        "install",
        os.path.dirname(__file__),
        "--output-folder=build",
        "--build=missing",
        "-s",
        "compiler.cppstd=17",
        "-s:h",
        "arch=x86_64",
        "-s:b",
        "arch=x86_64",
        "-s",
        f"build_type={'Debug' if DEBUG_MODE else 'Release'}",
        "-v",
        "debug",
    ]
)

CONAN_TOOLCHAIN_FILE = os.path.join(BUILD_DIR, "conan_toolchain.cmake")

cmake_args = []
cmake_args.append(f"-DCMAKE_TOOLCHAIN_FILE={CONAN_TOOLCHAIN_FILE}")
# try to be as verbose as possible
cmake_args.append(f"-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON")
cmake_args.append(f"-DCMAKE_MESSAGE_LOG_LEVEL=STATUS")
cmake_args.append(f"-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")

if DEBUG_MODE:
    cmake_args.append(f"-DCMAKE_BUILD_TYPE=Debug")

if platform.system() == "Windows":
    cmake_args.append(f"-DCMAKE_POLICY_DEFAULT_CMP0091=NEW")
    if platform.architecture()[0] == "32bit":
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
    author_email="qiskit@us.ibm.com",
    license="Apache 2.0",
    classifiers=classifiers,
    python_requires=">=3.7",
    install_requires=requirements,
    include_package_data=False,
    package_data={"qiskit_aer": ["VERSION.txt"], "qiskit_aer.library": ["*.csv"]},
    cmake_args=cmake_args,
    keywords="qiskit, simulator, quantum computing, backend",
    zip_safe=False,
)
