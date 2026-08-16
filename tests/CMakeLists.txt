# Catch2 is the only external dependency, and it is pulled in with FetchContent
# so that no system-wide package is needed on any of the supported platforms.
include(FetchContent)

set(STATISTICALC_CATCH2_VERSION "v3.7.1" CACHE STRING "Catch2 git tag to fetch")

# Keep Catch2 out of the install set and out of the developer warnings.
set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG ${STATISTICALC_CATCH2_VERSION}
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE)

FetchContent_MakeAvailable(Catch2)

add_executable(statisticalc_tests
  test_descriptive.cpp
  test_window.cpp
  test_distributions.cpp
  test_inference.cpp
  test_two_sample.cpp
  test_usability.cpp)

target_link_libraries(statisticalc_tests
  PRIVATE statisticalc::statisticalc Catch2::Catch2WithMain)

target_compile_features(statisticalc_tests PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(statisticalc_tests PRIVATE /W4 /bigobj)
else()
  target_compile_options(statisticalc_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(statisticalc_tests)
