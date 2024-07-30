cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(e3nn_cpp VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add the include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add the main library target
file(GLOB_RECURSE SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
add_library(e3nn_cpp ${SOURCES})

# Enable testing
enable_testing()

# Find Google Test
find_package(GTest REQUIRED)

# Add the tests
file(GLOB_RECURSE TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)
add_executable(e3nn_cpp_tests ${TEST_SOURCES})
target_link_libraries(e3nn_cpp_tests 
    e3nn_cpp 
    GTest::GTest 
    GTest::Main
)

# Discover tests
include(GoogleTest)
gtest_discover_tests(e3nn_cpp_tests)

# Print project structure for verification
message("Project Structure:")
message("  Sources: ${SOURCES}")
message("  Test Sources: ${TEST_SOURCES}")