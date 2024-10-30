import os
from conans import ConanFile, CMake

def version_name():
    build_version = os.getenv("BUILD_VERSION")

    if build_version == None:
        return "develop"
    else:
        return str(build_version)

class ClusterFunk(ConanFile):
    name = "ClusterFunk"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "coverage": [True, False]}
    default_options = {"shared": False, "coverage": False}
    generators = "cmake"
    exports_sources = {"CMakeLists.txt", "src/**", "test/**", "config/**"}
    keep_imports = True

    def requirements(self):
        pass

    def set_version(self):
        self.version = version_name()
        print("set_version:  ", self.version)

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_folder="")
        cmake.build()
        cmake.parallel = False
        cmake.test(output_on_failure=True)

    def package(self):
        self.copy("*.h", dst="include", src="src")
        self.copy("*.a", dst="lib", keep_path=False)

    def deploy(self):
        self.copy("*", dst="include", src="include")
        self.copy("*", dst="lib", src="lib")

    def package_info(self):
        self.cpp_info.libs = ["lib"]
