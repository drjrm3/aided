// primitives_bindings.cpp
//
// This file is used to create python bindings for the C++ functions in the
// primitives.h file. It uses pybind11 to create the bindings and
// exposes the gpow function to Python.
//
// Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
#include <iostream>
#include <vector>
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "primitives.h"
#include <pybind_utils.h>

#define DISPATCH(TX, TE) return dispatch_gpow<TX, TE>(x, e, x_is_array, e_is_array);

namespace aided {

// Type aware dispatching for gpow. This calls the associated function depending
// on the type of the arguments and whether they are arrays or scalars.
template<typename T, typename E>
pybind11::object dispatch_gpow(pybind11::object x,
                               pybind11::object e,
                               bool x_is_array,
                               bool e_is_array)
{
    if (!x_is_array && !e_is_array)
    {
        return pybind11::cast(gpow(x.cast<T>(), e.cast<E>()));
    }
    else if (x_is_array && !e_is_array)
    {
        const auto vec = gpow(x.cast<std::vector<T>>(), e.cast<E>());
        return vector_to_ndarray(vec);
    }
    else if (!x_is_array && e_is_array)
    {
        const auto vec = gpow(x.cast<T>(), e.cast<std::vector<E>>());
        return vector_to_ndarray(vec);
    }
    else
    {
        const auto vec = gpow(x.cast<std::vector<T>>(), e.cast<std::vector<E>>());
        return vector_to_ndarray(vec);
    }
}

// Function to handle the gpow operation. Relies on type aware dispatching.
pybind11::object gpow_py(pybind11::object x, pybind11::object e) {
    bool x_is_array = pybind11::isinstance<pybind11::list>(x) ||
                      pybind11::isinstance<pybind11::array>(x);
    bool e_is_array = pybind11::isinstance<pybind11::list>(e) ||
                      pybind11::isinstance<pybind11::array>(e);

    PyKind x_kind = py_type_signature(x, x_is_array);
    PyKind e_kind = py_type_signature(e, e_is_array);

    // clang-format off
    if (x_kind == PyKind::Float32 && e_kind == PyKind::Int32)   DISPATCH(float, int)
    if (x_kind == PyKind::Float64 && e_kind == PyKind::Int32)   DISPATCH(double, int)
    if (x_kind == PyKind::Float64 && e_kind == PyKind::Int64)   DISPATCH(double, int)
    if (x_kind == PyKind::Int32   && e_kind == PyKind::Int32)   DISPATCH(int, int)
    if (x_kind == PyKind::Int32   && e_kind == PyKind::Int64)   DISPATCH(int, int64_t)
    if (x_kind == PyKind::Int64   && e_kind == PyKind::Int32)   DISPATCH(int64_t, int)
    // clang-format on

    std::cerr << "[gpow] Unsupported type combination: "
          << "x_kind = " << py_kind_to_string(x_kind)
          << ", e_kind = " << py_kind_to_string(e_kind)
          << std::endl;

    throw std::runtime_error("Unsupported type combination for gpow.");
}

PYBIND11_MODULE(_primitives, m) {
    m.def("gpow", &gpow_py, "Custom power function for scalars and vectors");
}
} // namespace aided
