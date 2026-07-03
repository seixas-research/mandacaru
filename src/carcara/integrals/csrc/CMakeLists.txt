# file: csrc/CMakeLists.txt
#
# Builds libcarcara_integrals as a shared library with OpenMP.
#
# Standalone:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
# The resulting library lands in csrc/build/ where _backend.py looks for it.
#
# macOS note: Apple Clang needs Homebrew libomp:
#   brew install libomp
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
#         -DOpenMP_ROOT=$(brew --prefix libomp)

cmake_minimum_required(VERSION 3.18)
project(carcara_integrals C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math -funroll-loops")

add_library(carcara_integrals SHARED carcara_integrals.c)
target_include_directories(carcara_integrals PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(carcara_integrals PUBLIC OpenMP::OpenMP_C)
else()
    message(WARNING "OpenMP not found: building a serial fallback library.")
endif()

# Keep the output name stable and put it next to the sources' build/ dir.
set_target_properties(carcara_integrals PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    C_VISIBILITY_PRESET default)
