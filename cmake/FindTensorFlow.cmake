# FindTensorFlow.cmake
# Finds the TensorFlow library
#
# This will define the following variables:
#   TensorFlow_FOUND        - True if the system has TensorFlow
#   TensorFlow_INCLUDE_DIRS - TensorFlow include directory
#   TensorFlow_LIBRARIES    - TensorFlow libraries
#   TensorFlow_VERSION      - TensorFlow version

# Attempt to find TensorFlow in standard locations
find_path(TensorFlow_INCLUDE_DIR
  NAMES tensorflow/c/c_api.h
  PATHS
    /usr/local/include
    /usr/include
    /opt/homebrew/include
    /opt/local/include
    $ENV{TENSORFLOW_ROOT}/include
    $ENV{HOME}/tensorflow
    $ENV{HOME}/.local/include
)

find_library(TensorFlow_LIBRARY
  NAMES tensorflow tensorflow_cc tensorflow_framework
  PATHS
    /usr/local/lib
    /usr/lib
    /usr/local/lib64
    /usr/lib64
    /opt/homebrew/lib
    /opt/local/lib
    $ENV{TENSORFLOW_ROOT}/lib
    $ENV{HOME}/tensorflow/lib
    $ENV{HOME}/.local/lib
)

# Handle the QUIETLY and REQUIRED arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorFlow
  REQUIRED_VARS
    TensorFlow_LIBRARY
    TensorFlow_INCLUDE_DIR
)

# Set result variables
if(TensorFlow_FOUND)
  set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
  set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR})
endif()

mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)
