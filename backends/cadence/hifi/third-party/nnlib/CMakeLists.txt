include(ExternalProject)
cmake_minimum_required(VERSION 3.10.0)
project(cadence_nnlib)

set(CMAKE_C_COMPILER xt-clang)
set(CMAKE_CXX_COMPILER xt-clang++)
set(CMAKE_LINKER xt-ld)
  
        
add_custom_target( nnlib_target ALL COMMAND make install_nnlib -f makefile -C ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/xa_nnlib/build -j )

add_library(xa_nnlib STATIC IMPORTED GLOBAL)
add_dependencies(xa_nnlib nnlib_target)

set_property(
  TARGET xa_nnlib
  PROPERTY 
  IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/nnlib-hifi4/xa_nnlib/build/libxa_nnlib.a"
)

#set(_common_include_directories ${EXECUTORCH_ROOT}/..)
#install(
#  TARGETS xa_nnlib
#  DESTINATION lib
#  DESTINATION ${_common_include_directories}
#)









