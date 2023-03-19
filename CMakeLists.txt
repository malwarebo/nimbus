cmake_minimum_required(VERSION 3.5)

project(weather-forecasting LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(TensorFlow REQUIRED)

add_library(weather_model weather_model.cpp)
target_include_directories(weather_model PUBLIC ${TensorFlow_INCLUDE_DIRS})
target_link_libraries(weather_model ${TensorFlow_LIBRARIES})

add_executable(weather_forecasting main.cpp)
target_link(weather_forecasting weather_model)
target_include_directories(weather_forecasting PUBLIC ${TensorFlow_INCLUDE_DIRS})
target_link_libraries(weather_forecasting weather_model ${TensorFlow_LIBRARIES})

