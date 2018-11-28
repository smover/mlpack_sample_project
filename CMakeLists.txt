project(LSTM)
cmake_minimum_required(VERSION 3.0)

# The version number.
set (LSTM_VERSION_MAJOR 1)
set (LSTM_VERSION_MINOR 0)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


#############################################################################
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/src)

# TODO:
# add_subdirectory(src/learning)

# TODO
# Check for the mlpack library

# add the executable
add_executable(lstm ${PROJECT_SOURCE_DIR}/src/main.cpp)

