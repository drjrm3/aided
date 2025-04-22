// pybind_utils.h
//
// This file is used for most utility functions for pybind11 in aided.
//
// Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace aided {

// Enum class to represent the kind of data in a py::array.
enum class PyKind {
    Int32,
    Int64,
    Float32,
    Float64,
    Unknown
};

// Get the type signature of a Python object. This is used to determine the type
// of the object and whether it is an array or a scalar. The function returns
// a PyKind enum value that indicates the type of the object.
inline PyKind py_type_signature(pybind11::object obj, bool is_array=false)
{
    if (!is_array)
    {
        if (pybind11::hasattr(obj, "dtype"))
        {
            pybind11::object dtype = obj.attr("dtype");
            auto dtype_name = std::string(pybind11::str(dtype));
            if (dtype_name.find("int64") != std::string::npos)
                return PyKind::Int64;
            if (dtype_name.find("int32") != std::string::npos)
                return PyKind::Int32;
            if (dtype_name.find("float64") != std::string::npos)
                return PyKind::Float64;
            if (dtype_name.find("float32") != std::string::npos)
                return PyKind::Float32;
        }

        if (pybind11::isinstance<pybind11::float_>(obj)) return PyKind::Float64;
        if (pybind11::isinstance<pybind11::int_>(obj)) return PyKind::Int32;
    }
    else
    {
        if (pybind11::isinstance<pybind11::array>(obj))
        {
            pybind11::array arr = obj.cast<pybind11::array>();
            auto dtype = arr.dtype();
            auto kind = dtype.kind();
            int itemsize = dtype.itemsize();

            if (kind == 'f')
                return itemsize == 4 ? PyKind::Float32 : PyKind::Float64;
            else if (kind == 'i')
                return itemsize == 4 ? PyKind::Int32 : PyKind::Int64;
        }
        else if (pybind11::isinstance<pybind11::list>(obj))
        {
            pybind11::list lst = obj.cast<pybind11::list>();
            if (lst.empty()) return PyKind::Unknown;

            pybind11::object first = lst[0];
            if (pybind11::isinstance<pybind11::float_>(first)) return PyKind::Float64;
            if (pybind11::isinstance<pybind11::int_>(first)) return PyKind::Int32;
        }
    }

    return PyKind::Unknown;
}

// Convert a std::vector<T> to a py::array_t<T>
template<typename T>
pybind11::array_t<T> vector_to_ndarray(const std::vector<T> &vec) {
    // Create a py::array_t<T> with the same size as the vector.
    pybind11::array_t<T> arr(vec.size());

    // Get a pointer to the underlying data of the py::array_t<T>.
    pybind11::buffer_info info = arr.request();

    // Get a pointer to the underlying data of the arr.
    T* arr_ptr = static_cast<T*>(info.ptr);

    // Copy the data from the vector to the py::array_t<T>.
    std::memcpy(arr_ptr, vec.data(), vec.size() * sizeof(T));

    // Return the py::array_t<T>.
    return arr;
}

// Convert a PyKind enum to a string representation.
inline const char* py_kind_to_string(PyKind kind)
{
    switch (kind)
    {
    case PyKind::Int32:   return "Int32";
    case PyKind::Int64:   return "Int64";
    case PyKind::Float32: return "Float32";
    case PyKind::Float64: return "Float64";
    default:              return "Unknown";
    }
}

} // aided
