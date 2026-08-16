cmake_minimum_required(VERSION 3.16)

project(statisticalc
  VERSION 1.0.0
  DESCRIPTION "Header-only C++20 running statistics: descriptive and inferential"
  HOMEPAGE_URL "https://github.com/pbosetti/statisticalc"
  LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}" STATISTICALC_IS_TOP_LEVEL)

option(STATISTICALC_BUILD_TESTS    "Build the Catch2 test suite" ${STATISTICALC_IS_TOP_LEVEL})
option(STATISTICALC_BUILD_EXAMPLES "Build the examples"          ${STATISTICALC_IS_TOP_LEVEL})
option(STATISTICALC_INSTALL        "Generate the install target" ${STATISTICALC_IS_TOP_LEVEL})

# ---------------------------------------------------------------------------
# The library: header only, no dependency but the standard library
# ---------------------------------------------------------------------------
add_library(statisticalc INTERFACE)
add_library(statisticalc::statisticalc ALIAS statisticalc)

target_compile_features(statisticalc INTERFACE cxx_std_20)

target_include_directories(statisticalc INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# MSVC needs to be told to report the real value of __cplusplus and to keep
# the standard preprocessor/maths defines.
if(MSVC)
  target_compile_options(statisticalc INTERFACE /Zc:__cplusplus /permissive-)
  target_compile_definitions(statisticalc INTERFACE _USE_MATH_DEFINES NOMINMAX)
endif()

if(STATISTICALC_IS_TOP_LEVEL)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# ---------------------------------------------------------------------------
# Tests and examples
# ---------------------------------------------------------------------------
if(STATISTICALC_BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()

if(STATISTICALC_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# Installation (headers + CMake package config, so that find_package works)
# ---------------------------------------------------------------------------
if(STATISTICALC_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  install(DIRECTORY include/statisticalc
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  install(TARGETS statisticalc EXPORT statisticalcTargets)

  install(EXPORT statisticalcTargets
    FILE statisticalcTargets.cmake
    NAMESPACE statisticalc::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/statisticalc)

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/statisticalcConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT)

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/statisticalcConfig.cmake"
"include(CMakeFindDependencyMacro)\n\
include(\"\${CMAKE_CURRENT_LIST_DIR}/statisticalcTargets.cmake\")\n")

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/statisticalcConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/statisticalcConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/statisticalc)
endif()
