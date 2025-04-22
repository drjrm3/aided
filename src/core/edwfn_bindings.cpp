// edwfn_bindings.cpp
//
// This file is used to create python bindings for the C++ functions in the
// primitives.h file. It uses pybind11 to create the bindings and
// exposes the gpow function to Python.
//
// Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.

#include <cmath>

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "edwfn.h"
#include <pybind_utils.h>


namespace aided {

#define DISPATCH(T)                            \
    return dispatch_gen_chi<T>(x,              \
                               y,              \
                               z,              \
                               ider,           \
                               last_der,       \
                               last_point,     \
                               types,          \
                               centers,        \
                               expons,         \
                               atpos,          \
                               chi,            \
                               chi1,           \
                               chi2);

template<typename T>
std::tuple<bool, pybind11::object, pybind11::object>
dispatch_gen_chi(pybind11::object x,
                pybind11::object y,
                pybind11::object z,
                pybind11::object ider,
                pybind11::object& last_der,
                pybind11::object& last_point,
                const pybind11::array& types,
                const pybind11::array& centers,
                const pybind11::array& expons,
                const pybind11::array& atpos,
                pybind11::array& chi,
                pybind11::array& chi1,
                pybind11::array& chi2
)
{
    // Convert pass by reference arguments to lvalue types.
    auto _last_dir = pybind11::cast<int32_t>(last_der);
    auto _chi = chi.mutable_unchecked<T, 1>();
    auto _chi1 = chi1.mutable_unchecked<T, 2>();
    auto _chi2 = chi2.mutable_unchecked<T, 2>();
    const auto _atpos = pybind11::cast<std::vector<T>>(atpos.attr("flatten")());

    std::vector<T> _last_point;
    if (last_point.is_none()) {
        // If last_point is None, create a sentinel vector with NaN values.
        _last_point = std::vector<T>{
            std::numeric_limits<T>::quiet_NaN(),
            std::numeric_limits<T>::quiet_NaN(),
            std::numeric_limits<T>::quiet_NaN()
        };
    } else {
        // Optionally, you could check if it's actually an iterable with three elements.
        // For now, we assume it is.
        _last_point = pybind11::cast<std::vector<T>>(last_point);
    }

    auto computed_chi = gen_chi<T>(
        x.cast<T>(),
        y.cast<T>(),
        z.cast<T>(),
        ider.cast<int32_t>(),
        _last_point,
        _last_dir,
        pybind11::cast<std::vector<int32_t>>(types),
        pybind11::cast<std::vector<int32_t>>(centers),
        pybind11::cast<std::vector<T>>(expons),
        _atpos,
        static_cast<T*>(chi.mutable_data()), chi.size(),
        static_cast<T*>(chi1.mutable_data()), chi1.shape(0), chi1.shape(1),
        static_cast<T*>(chi2.mutable_data()), chi2.shape(0), chi2.shape(1)
    );

    // Copy contents back to the pybind11 objects
    last_point = pybind11::cast(_last_point);
    last_der = pybind11::cast(_last_dir);

    auto output = std::make_tuple(computed_chi, last_point, last_der);

    return output;
}

// Function to handle the gpow operation. Relies on type aware dispatching.
std::tuple<bool, pybind11::object, pybind11::object>
gen_chi_py(pybind11::object x,
                pybind11::object y,
                pybind11::object z,
                pybind11::object ider,
                pybind11::object& last_der,
                pybind11::object& last_point,
                const pybind11::array& types,
                const pybind11::array& centers,
                const pybind11::array& expons,
                const pybind11::array& atpos,
                pybind11::array& chi,
                pybind11::array& chi1,
                pybind11::array& chi2
)
{

    // Print out the Python type of x:
    PyKind x_kind = py_type_signature(x);


    if (x_kind == PyKind::Float64) DISPATCH(double)
    if (x_kind == PyKind::Float32) DISPATCH(float)

    throw std::runtime_error("Unsupported type for x");

}

PYBIND11_MODULE(_edwfn, m) {
    m.def("gen_chi", &gen_chi_py, "Generation of chi.");
}
} // namespace aided
