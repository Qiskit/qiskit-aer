from setuptools import build_meta as _orig
from packaging import version as _version
from packaging.tags import sys_tags as _sys_tags
from skbuild.exceptions import SKBuildError as _SKBuildError
from skbuild.cmaker import get_cmake_version as _get_cmake_version
import subprocess as _subprocess
import platform as _platform

prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
build_wheel = _orig.build_wheel
build_sdist = _orig.build_sdist
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist


def _cmake_required():
    try:
        version = _version.parse(_get_cmake_version())
        if version in {_version.parse("3.17.0"), _version.parse("3.17.1")}:
            return True
        if version >= _version.parse("3.8"):
            print("Using System version of cmake")
            return False
    except _SKBuildError:
        pass

    return True


def _ninja_required():
    if _platform.system() == "Windows":
        print("Ninja is part of the MSVC installation on Windows")
        return False

    for generator in ("ninja", "make"):
        try:
            _subprocess.check_output([generator, "--version"])
            print(f"Using System version of {generator}")
            return False
        except (OSError, _subprocess.CalledProcessError):
            pass

    return True


def get_requires_for_build_wheel(self, config_settings=None):
    packages = []
    if _cmake_required():
        packages.append("cmake!=3.17.1,!=3.17.0")
    if _ninja_required():
        packages.append("ninja")

    return _orig.get_requires_for_build_wheel(config_settings) + packages
