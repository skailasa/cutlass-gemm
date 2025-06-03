add_executable(gemm main.cpp ${CPP_SOURCES})

target_include_directories(gemm
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gemm
    PRIVATE
        cutlass_kernels
)

add_custom_target(run_gemm
  COMMAND ${CMAKE_COMMAND} -E echo "🚀 Running GEMM..."
  COMMAND ./gemm
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples/gemm
  DEPENDS gemm
)
