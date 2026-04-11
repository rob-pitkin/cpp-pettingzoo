cmake_minimum_required(VERSION 3.15)
project(cpp_pettingzoo)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for linters/IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Find Python from the current environment
# Prefer virtual environment if one is active
if(DEFINED ENV{VIRTUAL_ENV})
    set(Python_ROOT_DIR $ENV{VIRTUAL_ENV})
    set(Python_FIND_VIRTUALENV ONLY)
    message(STATUS "Found virtual environment: $ENV{VIRTUAL_ENV}")
else()
    set(Python_FIND_VIRTUALENV FIRST)
    message(STATUS "No virtual environment detected, using system Python")
endif()
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Using Python: ${Python_EXECUTABLE}")

# Find pybind11 - will use Python's pybind11
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# Build pybind11 module for Simple
pybind11_add_module(_simple_core
    cpp_pettingzoo/simple/cpp/bindings.cpp
    cpp_pettingzoo/simple/cpp/simple_env.cpp
    cpp_pettingzoo/simple/cpp/simple_scenario.cpp
    core/world.cpp
    core/base_env.cpp
)

# Build pybind11 module for SimpleSpread
pybind11_add_module(_simple_spread
    cpp_pettingzoo/simple_spread/cpp/bindings.cpp
    cpp_pettingzoo/simple_spread/cpp/simple_spread_env.cpp
    cpp_pettingzoo/simple_spread/cpp/simple_spread_scenario.cpp
    core/world.cpp
    core/base_env.cpp
)

# Build pybind11 module for SimpleReference
pybind11_add_module(_simple_reference
    cpp_pettingzoo/simple_reference/cpp/bindings.cpp
    cpp_pettingzoo/simple_reference/cpp/simple_reference_env.cpp
    cpp_pettingzoo/simple_reference/cpp/simple_reference_scenario.cpp
    core/world.cpp
    core/base_env.cpp
)
