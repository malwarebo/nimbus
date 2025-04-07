cmake_minimum_required(VERSION 3.10)

project(weather-forecasting LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find TensorFlow package
# Using the Homebrew installation of libtensorflow
set(TENSORFLOW_ROOT "/opt/homebrew/Cellar/libtensorflow/2.19.0")
find_library(TensorFlow_LIBRARIES
  NAMES tensorflow
  PATHS "${TENSORFLOW_ROOT}/lib"
  REQUIRED)
find_path(TensorFlow_INCLUDE_DIRS
  NAMES tensorflow/c/c_api.h
  PATHS "${TENSORFLOW_ROOT}/include"
  REQUIRED)

# Print information for debugging
message(STATUS "TensorFlow libraries: ${TensorFlow_LIBRARIES}")
message(STATUS "TensorFlow include dirs: ${TensorFlow_INCLUDE_DIRS}")

# Include TensorFlow headers in all targets
include_directories(${TensorFlow_INCLUDE_DIRS})

# Add models library
add_library(models models/weather_model.cpp)
target_include_directories(models PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TensorFlow_INCLUDE_DIRS})
target_link_libraries(models ${TensorFlow_LIBRARIES})

# Add main executable
add_executable(nimbus main.cpp)
target_include_directories(nimbus PUBLIC ${TensorFlow_INCLUDE_DIRS})
target_link_libraries(nimbus models ${TensorFlow_LIBRARIES})

