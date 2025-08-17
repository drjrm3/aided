add_library(wfn OBJECT wfn.cpp)
target_link_libraries(wfn
    PUBLIC
    primitives
)

set_target_properties(wfn PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

pybind11_add_module(_wfn wfn_bindings.cpp $<TARGET_OBJECTS:wfn>)
target_link_libraries(_wfn
    PRIVATE
    wfn
    pybind_utils
    primitives
)

set_target_properties(_wfn PROPERTIES
    CXX_STANDARD 17
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/aided/math"
)

# Put this in place for testing.
set_target_properties(_wfn PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../../aided/core/ed"
)
