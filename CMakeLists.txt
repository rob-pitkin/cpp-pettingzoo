cmake_minimum_required(VERSION 3.15)
project(cpp_pettingzoo)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Find Python from the current environment
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Using Python: ${Python_EXECUTABLE}")

# Find pybind11 - will use Python's pybind11
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# Build simple_core library
add_library(simple_core
    cpp_pettingzoo/simple/cpp/simple_core.cpp
)

# Build pybind11 module
pybind11_add_module(_simple_core
    cpp_pettingzoo/simple/cpp/bindings.cpp
    cpp_pettingzoo/simple/cpp/simple_core.cpp
)

# Test executable (optional, for testing compilation)
add_executable(test_simple test_compile.cpp)
target_link_libraries(test_simple simple_core)
