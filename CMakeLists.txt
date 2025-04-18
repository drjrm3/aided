cmake_minimum_required(VERSION 3.10)

project(aided LANGUAGES CXX)


# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Add aggressive compiler flags to release builds
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math"
)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Force discovery of the correct pybind11
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED CONFIG)

add_subdirectory(src)

# Add specific targets
install(TARGETS _primitives
    LIBRARY DESTINATION aided/math
)
install(TARGETS _edwfn
    LIBRARY DESTINATION aided/core
)
