// edwfn.h
//
// Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
#include <cmath>
#include <stdexcept>

#include "edwfn.h"
#include <primitives.h>

// Generation of chi.
template <typename T>
bool gen_chi(
    T x,
    T y,
    T z,
    int32_t ider,
    std::vector<T>& lastPoint,           // Last position analyzed
    int32_t& lastDer,                    // Last derivative analyzed
    const std::vector<int32_t>& types,   // Gaussian types     (nprims,)
    const std::vector<int32_t>& centers, // Gaussian centers   (nprims,
    const std::vector<T>& expons,        // Gaussian exponents (nprims,)
    const std::vector<T>& _atpos,        // Atomic positions   (natoms, 3)
    T* chi, size_t nprims,               // Output: chi        (nprims,)
    T* chi1, size_t __r1, size_t __c1,   // Output: chi        (nprims, 3)
    T* chi2, size_t __r2, size_t __c2    // Output: chi        (nprims, 6)
)
{
    // Create simple lambda function that allos me to get atpos(i, j) as a 1D array.
    const auto atpos = [&_atpos](size_t i, size_t j) {
        return _atpos[i * 3 + j];
    };

    // Check for sizing consistencies.
    if (__r1 != nprims || __r2 != nprims)
        throw std::runtime_error(
            "gen_chi: chi1 and chi2 must be the same size as chi.");
    else if (__c1 != 3 || __c2 != 6)
        throw std::runtime_error(
            "gen_chi: chi1 must be (nprims, 3) and chi2 must be (nprims, 6).");

    // If this is the same as the last point, and the last derivative, return true.
    bool samePoint = x == lastPoint[0] &&
                     y == lastPoint[1] &&
                     z == lastPoint[2];
    if (samePoint && ider == lastDer)
        return false;

    // Update this point to be the last point.
    lastPoint = {x, y, z};
    lastDer = ider;

    // Calculate pxs, pys, pzs - the position of the Gaussian centers.
    for (size_t i = 0; i < nprims; ++i) {
        const auto px = x - atpos(centers[i], 0);
        const auto py = y - atpos(centers[i], 1);
        const auto pz = z - atpos(centers[i], 2);

        const auto px2 = px * px;
        const auto py2 = py * py;
        const auto pz2 = pz * pz;
        const auto alpha = expons[i];
        const auto arg = -alpha * (px2 + py2 + pz2);
        const auto expon = exp(arg);

        const auto l = LMNS[types[i]][0];
        const auto m = LMNS[types[i]][1];
        const auto n = LMNS[types[i]][2];

        const auto xl = gpow(px, l);
        const auto ym = gpow(py, m);
        const auto zn = gpow(pz, n);
        chi[i] = expon * xl * ym * zn;

        if (ider == 0) continue;

        // First derivative
        const auto twoa = 2.0 * alpha;

        const auto term11 = gpow(px, l - 1) * l;
        const auto term12 = gpow(py, m - 1) * m;
        const auto term13 = gpow(pz, n - 1) * n;

        const auto xyexp = xl * ym * expon;
        const auto xzexp = xl * zn * expon;
        const auto yzexp = ym * zn * expon;

        chi1[i * 3 + 0] = yzexp * (term11 - twoa * xl * px);
        chi1[i * 3 + 1] = xzexp * (term12 - twoa * ym * py);
        chi1[i * 3 + 2] = xyexp * (term13 - twoa * zn * pz);

        if (ider == 1) continue;

        const auto twoa_chi = twoa * chi[i];

        // xx, yy, zz
        chi2[i * 6 + 0] = gpow(px, l - 2) * yzexp * l * (l-1) - twoa_chi *
            (2.0 * l + 1.0 - twoa * px2);
        chi2[i * 6 + 3] = gpow(py, m - 2) * xzexp * m * (m-1) - twoa_chi *
            (2.0 * m + 1.0 - twoa * py2);
        chi2[i * 6 + 5] = gpow(pz, n - 2) * xyexp * n * (n-1) - twoa_chi *
            (2.0 * n + 1.0 - twoa * pz2);

        const auto expee = twoa * expon;
        const auto foura_two_chi = 4.0 * alpha * alpha * chi[i];

        // xy
        chi2[i * 6 + 1] = (
            term11 * term12 * zn * expon
            - term12 * xl * px * zn * expee
            - term11 * ym * py * zn * expee
            + px * py * foura_two_chi
        );
        //  xz
        chi2[i * 6 +  2] = (
            term11 * term13 * ym * expon
            - term13 * xl * px * ym * expee
            - term11 * zn * pz * ym * expee
            + px * pz * foura_two_chi
        );
        // yz
        chi2[i * 6 +  4] = (
            term12 * term13 * xl * expon
            - term13 * ym * py * xl * expee
            - term12 * zn * pz * xl * expee
            + py * pz * foura_two_chi
        );
    }

    return true;
}

template bool gen_chi<double>(
    double, double, double, int32_t,
    std::vector<double>&, int32_t&,
    const std::vector<int32_t>&, const std::vector<int32_t>&,
    const std::vector<double>&, const std::vector<double>&,
    double*, size_t,
    double*, size_t, size_t,
    double*, size_t, size_t
);

template bool gen_chi<float>(
    float, float, float, int32_t,
    std::vector<float>&, int32_t&,
    const std::vector<int32_t>&, const std::vector<int32_t>&,
    const std::vector<float>&, const std::vector<float>&,
    float*, size_t,
    float*, size_t, size_t,
    float*, size_t, size_t
);
