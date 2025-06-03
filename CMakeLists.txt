cmake_minimum_required(VERSION 3.19)
project(CUTLASS_PRACTICE LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)


# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 86)

# in debug mode, add debug symbols to device code
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/external/cutlass/include
)
# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
add_executable(cutlass_gemm cutlass_gemm.cu ${SRC})
set_target_properties(cutlass_gemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(cutlass_gemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

add_custom_target(run_cutlass_gemm
  COMMAND ${CMAKE_COMMAND} -E echo "🚀 Running GEMM With CUTLASS..."
  COMMAND ./cutlass_gemm
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DEPENDS cutlass_gemm
)