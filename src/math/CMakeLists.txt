add_library(primitives INTERFACE)
target_include_directories(primitives INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

pybind11_add_module(_primitives primitives_bindings.cpp)
target_link_libraries(_primitives PRIVATE primitives pybind_utils)

set_target_properties(_primitives PROPERTIES
    CXX_STANDARD 17
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/aided/math"
)

# Put this in place for testing.
set_target_properties(_primitives PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../aided/math"
)
