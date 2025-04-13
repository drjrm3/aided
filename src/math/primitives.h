// primitives.cpp
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <cmath>

// scalar-scalar
// Assumes that E is an integer!!
template<typename T, typename E>
auto gpow(T x, E e) -> std::enable_if_t<std::is_integral_v<E>, T>
{
    if (e == 0) return 1.0;
    if (x == 0) return 0.0;

    T result = 1.0;
    // Positive exponent
    if (e > 0) {
        for (E i = 0; i < e; ++i) result *= x;
        return result;
    }

    // Negative exponent
    for (E i = 0; i < -e; ++i) result *= x;
    return 1.0 / result;
}

// clang-format off
// vector-function
template<typename T, typename F>
auto gpow(const std::vector<T>& x, F f) 
    -> std::enable_if_t<!std::is_arithmetic_v<F>, std::vector<T>>
{
    std::vector<T> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) result[i] = gpow(x[i], f(i));
    return result;
}

// vector-scalar
template<class T, class E>
auto gpow(const std::vector<T>& x, E e) 
    -> std::enable_if_t<std::is_arithmetic_v<E>, std::vector<T>>
{
    std::vector<T> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) result[i] = gpow(x[i], e);
    return result;
}

// scalar-vector
template <typename T, typename E>
auto gpow(T x, const std::vector<E>& e) {
    std::vector<T> result(e.size());
    for (size_t i = 0; i < e.size(); ++i) result[i] = gpow(x, e[i]);
    return result;
}

// vector-vector
template <typename T, typename E>
auto gpow(const std::vector<T>& x, const std::vector<E>& e) {
    if (x.size() != e.size()) {
        throw std::runtime_error("Mismatched vector sizes");
    }
    std::vector<T> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) result[i] = gpow(x[i], e[i]);
    return result;
}
