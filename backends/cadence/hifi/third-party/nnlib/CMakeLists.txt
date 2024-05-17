
cmake_minimum_required(VERSION 3.10.0)
project(cadence_nnlib)

        
add_custom_target( nnlib_target ALL COMMAND make install_nnlib -f makefile -C ${EXECUTORCH_ROOT}/backends/cadence/hifi/third-party/nnlib/nnlib-hifi4/xa_nnlib/build -j )

add_library(xa_nnlib STATIC IMPORTED GLOBAL)
add_dependencies(xa_nnlib nnlib_target)

set_property(
  TARGET xa_nnlib
  PROPERTY 
  IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/nnlib-hifi4/xa_nnlib/build/xa_nnlib.a"
)











