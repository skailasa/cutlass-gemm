cmake_minimum_required(VERSION 3.20)

# Function to locate spack packages on system
function(find_spack_package name out_path)
  execute_process(COMMAND spack location -i ${name} OUTPUT_VARIABLE _path OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(${out_path} "${_path}" PARENT_SCOPE)
endfunction()

function(import_spack_library pkg_name target_name)
  find_spack_package(${pkg_name} PKG_PATH)
  file(GLOB LIBRARY "${PKG_PATH}/lib/*/lib${pkg_name}.so")
  if (NOT LIBRARY)
    file(GLOB LIBRARY "${PKG_PATH}/lib/lib${pkg_name}.so")
  endif()
  # Make sure that only one file is matched by extracting first hit
  list(GET LIBRARY 0 LIBRARY)

  set(INCLUDE_DIR "${PKG_PATH}/include")

  add_library(${target_name} SHARED IMPORTED)

  # Setting properties for a library that I'm not building, but CMake is fetching
  set_target_properties(${target_name} PROPERTIES
  IMPORTED_LOCATION ${LIBRARY}
  INTERFACE_INCLUDE_DIRECTORIES ${INCLUDE_DIR}
  )

endfunction()

project(cutlass_gemm LANGUAGES C CXX)

# Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_spack_package(cuda CUDA_PATH)
set(CUDAToolkit_ROOT "${CUDA_PATH}")
set(CMAKE_PREFIX_PATH "${CUDAToolkit_ROOT};${CMAKE_PREFIX_PATH}")
set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc")
find_package(CUDAToolkit REQUIRED)

# Specify sources
set(CPP_SOURCES
  ${CMAKE_SOURCE_DIR}/include/gemm_cutlass.hpp
  CACHE INTERNAL "C++ source files" # to be visible by lower level CMakeLists files
)

add_library(cutlass_kernels INTERFACE)
target_include_directories(cutlass_kernels INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/cutlass/include
)


# Custom commands
file(GLOB_RECURSE FORMAT_SOURCES
  "${CMAKE_SOURCE_DIR}/examples/*.cpp"
  "${CMAKE_SOURCE_DIR}/include/*.cpp"
  "${CMAKE_SOURCE_DIR}/include/*.hpp"
)

find_program(CLANG_FORMAT NAMES clang-format)
if(CLANG_FORMAT)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${FORMAT_SOURCES}
    COMMENT "🧼 Running clang-format..."
  )
else()
  message(WARNING "clang-format not found; skipping format target.")
endif()

add_subdirectory(examples)
