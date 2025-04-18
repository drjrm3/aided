add_library(edwfn OBJECT edwfn.cpp)
target_link_libraries(edwfn
    PUBLIC
    primitives
)

set_target_properties(edwfn PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

pybind11_add_module(_edwfn edwfn_bindings.cpp $<TARGET_OBJECTS:edwfn>)
target_link_libraries(_edwfn
    PRIVATE
    edwfn
    pybind_utils
    primitives
)

set_target_properties(_edwfn PROPERTIES
    CXX_STANDARD 17
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/aided/math"
)

# Put this in place for testing.
set_target_properties(_edwfn PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../aided/core"
)
