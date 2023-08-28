cmake_minimum_required(VERSION 3.5)

project(weather-forecasting LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(TensorFlow REQUIRED)

add_library(models weather_model.cpp)
target_include_directories(models PUBLIC ${TensorFlow_INCLUDE_DIRS})
target_link_libraries(models ${TensorFlow_LIBRARIES})

add_executable(nimbus main.cpp)
target_link(nimbus models)
target_include_directories(nimbus PUBLIC ${TensorFlow_INCLUDE_DIRS})
target_link_libraries(nimbus models ${TensorFlow_LIBRARIES})

