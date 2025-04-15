// primitives.cpp
#include <vector>
#include <cmath>

// scalar-scalar
template <typename T, typename E>
T gpow_scalar(T x, E e) {
    if (e == 0) return 1.0;
    if (x == 0) return 0.0;
    return std::pow(x, e);
}

// vector-scalar
template <typename T, typename E>
std::vector<T> gpow(const std::vector<T>& x, E e) {
    std::vector<T> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = gpow_scalar(x[i], e);
    }
    return result;
}

// scalar-vector
template <typename T, typename E>
std::vector<T> gpow(T x, const std::vector<E>& e) {
    std::vector<T> result(e.size());
    for (size_t i = 0; i < e.size(); ++i) {
        result[i] = gpow_scalar(x, e[i]);
    }
    return result;
}

// vector-vector
template <typename T, typename E>
std::vector<T> gpow(const std::vector<T>& x, const std::vector<E>& e) {
    if (x.size() != e.size()) {
        throw std::runtime_error("Mismatched vector sizes");
    }
    std::vector<T> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = gpow_scalar(x[i], e[i]);
    }
    return result;
}
