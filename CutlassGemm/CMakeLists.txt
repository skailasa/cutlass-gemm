add_library(CutlassGemm INTERFACE)

# Add local headers
target_include_directories(CutlassGemm
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Add cutlass headers
target_include_directories(CutlassGemm
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cutlass/include>
)