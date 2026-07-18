# file: csrc/CMakeLists.txt
#
# Builds libcarcara_integrals as a shared library with OpenMP, on both
# Linux and macOS.
#
# Standalone:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
# The resulting library lands in csrc/build/ where _backend.py looks for it.
#
# Linux (GCC or Clang): OpenMP is found out of the box.
# macOS (Apple Clang): install Homebrew libomp once (`brew install libomp`);
#   this file then auto-discovers it via `brew --prefix libomp`, so no manual
#   flags are needed.  You may still override with -DOpenMP_ROOT=<path>.

cmake_minimum_required(VERSION 3.18)
project(carcara_integrals C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O3 -ffast-math -funroll-loops")

# On macOS, dependent shared libraries (libomp) are located at runtime via
# @rpath; keep the install name relative so the .dylib is relocatable.
if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()

add_library(carcara_integrals SHARED carcara_integrals.c)
target_include_directories(carcara_integrals PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# --- OpenMP -----------------------------------------------------------------
# Apple Clang ships without the OpenMP runtime, so find_package(OpenMP) usually
# fails on macOS.  Seed the hint variables from Homebrew's libomp before the
# find so the standard path succeeds; fall back to wiring libomp by hand.
if(APPLE AND NOT DEFINED OpenMP_ROOT)
    find_program(BREW_EXECUTABLE brew)
    if(BREW_EXECUTABLE)
        execute_process(COMMAND ${BREW_EXECUTABLE} --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
    endif()
    if(LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}")
        set(OpenMP_ROOT "${LIBOMP_PREFIX}")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(carcara_integrals PUBLIC OpenMP::OpenMP_C)
elseif(APPLE AND LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
    # Manual libomp wiring when the CMake module still could not assemble the
    # imported target (older CMake / non-standard toolchains).
    target_compile_options(carcara_integrals PRIVATE -Xpreprocessor -fopenmp)
    target_include_directories(carcara_integrals PRIVATE "${LIBOMP_PREFIX}/include")
    target_link_libraries(carcara_integrals PUBLIC "${LIBOMP_PREFIX}/lib/libomp.dylib")
    message(STATUS "OpenMP wired manually via Homebrew libomp at ${LIBOMP_PREFIX}")
else()
    message(WARNING "OpenMP not found: building a serial fallback library.")
endif()

# Keep the output name stable and put it next to the sources' build/ dir.
set_target_properties(carcara_integrals PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    C_VISIBILITY_PRESET default)
