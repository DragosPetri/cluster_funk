from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout
from conan.tools.files import copy

class ClusterFunk(ConanFile):
    name = "ClusterFunk"
    settings = "os", "compiler", "build_type", "arch"
    generators = {"CMakeDeps", "CMakeToolchain"}
    exports_sources = {"CMakeLists.txt", "src/**", "test/**", "config/**"}
    keep_imports = True

    def requirements(self):
        pass

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

        if not self.conf.get("tools.build:skip_test", check_type=bool):
            cmake.test()

    def package(self):
        copy(self, "*.h", dst="include", src="src")
        copy(self, "*.a", dst="lib", keep_path=False)

    def deploy(self):
        copy(self, "*", self.package_folder, self.deploy_folder)

    def layout(self):
        cmake_layout(self, build_folder="build")
