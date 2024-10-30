cmake_minimum_required(VERSION 3.29)
project(cluster_funk VERSION 0.0.1)

#-------------------------------------------------------------------------------
# Compiler config
#-------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(
    -Wall
    -Wextra
    -Wnon-virtual-dtor
    -Wunused
    -Wnull-dereference
    -fdiagnostics-color=always
)

#-------------------------------------------------------------------------------
# Conan config
#-------------------------------------------------------------------------------

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#-------------------------------------------------------------------------------
# Add dependencies
#-------------------------------------------------------------------------------

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/app)

#-------------------------------------------------------------------------------
# Create libraries
#-------------------------------------------------------------------------------

file(GLOB_RECURSE LIB_SOURCES
    "./src/data_objects/*.h"
    "./src/algebra/*.h"
    "./src/gmm_em/*.h"
    "./src/kmeans/*.h"
)

#-------------------------------------------------------------------------------
# Create application
#-------------------------------------------------------------------------------

add_executable(
    clustering_cli
    ${PROJECT_SOURCE_DIR}/app/clustering_cli.cpp
    ${LIB_SOURCES}
)

#-------------------------------------------------------------------------------
# Create test application
#-------------------------------------------------------------------------------

file(GLOB TEST_SOURCES
    "./test/csvcpp/*.h"
    "./test/csvcpp/*.cpp"
    "./test/acutest/*.h"
    "./test/acutest/*.cpp"
    "./test/*.h"
    "./test/*.cpp"
)

add_executable(clustering_prototyping_test ${TEST_SOURCES} ${LIB_SOURCES})

#-------------------------------------------------------------------------------
# Include target libraries
#-------------------------------------------------------------------------------

include(Dart)
enable_testing()
set(CMAKE_CTEST_ARGUMENTS "-VV")

add_test(
  NAME clustering_prototyping_test
  COMMAND $<TARGET_FILE:clustering_prototyping_test> -vv -t
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/test"
)
