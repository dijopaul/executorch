# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Set the minimum required version of CMake for this project.
cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Set the project name.
project(cadence_backend)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)
set(TARGET_DIR reference)

if(EXECUTORCH_NNLIB_OPT)
  set(TARGET_DIR hifi)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_DIR}/third-party/nnlib)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Find prebuilt libraries. executorch package should contain portable_ops_lib,

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_DIR}/operators)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_DIR}/kernels)
