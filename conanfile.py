from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps

class AerConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("openblas/0.3.30")
        self.requires("nlohmann_json/3.11.3")
        self.requires("spdlog/1.13.0")
        self.requires("fmt/10.2.1")

        if self.settings.os in ["Linux", "Macos"]:
            self.requires("llvm-openmp/20.1.6")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["CMAKE_POLICY_DEFAULT_CMP0091"] = "NEW"
        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()