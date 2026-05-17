cmake_minimum_required(VERSION 3.15)
project(cpp_pettingzoo)

# Default to Release if no build type specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

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

# Build pybind11 module for SimpleSpeakerListener
pybind11_add_module(_simple_speaker_listener
    cpp_pettingzoo/simple_speaker_listener/cpp/bindings.cpp
    cpp_pettingzoo/simple_speaker_listener/cpp/simple_speaker_listener_env.cpp
    cpp_pettingzoo/simple_speaker_listener/cpp/simple_speaker_listener_scenario.cpp
    core/world.cpp
    core/base_env.cpp
)

# Build pybind11 module for SimpleAdversary
pybind11_add_module(_simple_adversary
    cpp_pettingzoo/simple_adversary/cpp/bindings.cpp
    cpp_pettingzoo/simple_adversary/cpp/simple_adversary_env.cpp
    cpp_pettingzoo/simple_adversary/cpp/simple_adversary_scenario.cpp
    core/world.cpp
    core/base_env.cpp
)

# Build pybind11 module for SimpleFormation
pybind11_add_module(_simple_formation
    cpp_pettingzoo/simple_formation/cpp/bindings.cpp
    cpp_pettingzoo/simple_formation/cpp/simple_formation_env.cpp
    cpp_pettingzoo/simple_formation/cpp/simple_formation_scenario.cpp
    core/world.cpp
    core/base_env.cpp
)

# Build pybind11 module for SimplePush
pybind11_add_module(_simple_push
    cpp_pettingzoo/simple_push/cpp/bindings.cpp
    cpp_pettingzoo/simple_push/cpp/simple_push_env.cpp
    cpp_pettingzoo/simple_push/cpp/simple_push_scenario.cpp
    core/world.cpp
    core/base_env.cpp
)

# Build pybind11 module for SimpleTag
pybind11_add_module(_simple_tag
    cpp_pettingzoo/simple_tag/cpp/bindings.cpp
    cpp_pettingzoo/simple_tag/cpp/simple_tag_env.cpp
    cpp_pettingzoo/simple_tag/cpp/simple_tag_scenario.cpp
    core/world.cpp
    core/base_env.cpp
)

# Build pybind11 module for SimpleLine
pybind11_add_module(_simple_line
    cpp_pettingzoo/simple_line/cpp/bindings.cpp
    cpp_pettingzoo/simple_line/cpp/simple_line_env.cpp
    cpp_pettingzoo/simple_line/cpp/simple_line_scenario.cpp
    core/world.cpp
    core/base_env.cpp
)

# Build pybind11 module for CollectTreasure
pybind11_add_module(_collect_treasure
    cpp_pettingzoo/collect_treasure/cpp/bindings.cpp
    cpp_pettingzoo/collect_treasure/cpp/collect_treasure_env.cpp
    cpp_pettingzoo/collect_treasure/cpp/collect_treasure_scenario.cpp
    core/world.cpp
    core/base_env.cpp
)

# Build pybind11 module for SimpleWorldComm
pybind11_add_module(_simple_world_comm
    cpp_pettingzoo/simple_world_comm/cpp/bindings.cpp
    cpp_pettingzoo/simple_world_comm/cpp/simple_world_comm_env.cpp
    cpp_pettingzoo/simple_world_comm/cpp/simple_world_comm_scenario.cpp
    core/world.cpp
    core/base_env.cpp
)

# Build pybind11 module for SimpleCrypto
pybind11_add_module(_simple_crypto
    cpp_pettingzoo/simple_crypto/cpp/bindings.cpp
    cpp_pettingzoo/simple_crypto/cpp/simple_crypto_env.cpp
    cpp_pettingzoo/simple_crypto/cpp/simple_crypto_scenario.cpp
    core/world.cpp
    core/base_env.cpp
)
