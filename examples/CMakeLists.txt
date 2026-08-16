add_executable(statisticalc_example example.cpp)
target_link_libraries(statisticalc_example PRIVATE statisticalc::statisticalc)
target_compile_features(statisticalc_example PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(statisticalc_example PRIVATE /W4)
else()
  target_compile_options(statisticalc_example PRIVATE -Wall -Wextra -Wpedantic)
endif()
