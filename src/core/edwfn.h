// edwfn.h
//
// Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
#pragma once
#include <vector>
#include <cmath>
#include <array>

// number of primitives
constexpr size_t nLMNS = 35;

// LMNS[i] is the triple (l,m,n) for primitive i
constexpr std::array<std::array<int32_t,3>, nLMNS> LMNS = {{
    {{0,0,0}}, {{1,0,0}}, {{0,1,0}}, {{0,0,1}}, {{2,0,0}},
    {{0,2,0}}, {{0,0,2}}, {{1,1,0}}, {{1,0,1}}, {{0,1,1}},
    {{3,0,0}}, {{0,3,0}}, {{0,0,3}}, {{2,1,0}}, {{2,0,1}},
    {{0,2,1}}, {{1,2,0}}, {{1,0,2}}, {{0,1,2}}, {{1,1,1}},
    {{4,0,0}}, {{0,4,0}}, {{0,0,4}}, {{3,1,0}}, {{3,0,1}},
    {{1,3,0}}, {{0,3,1}}, {{1,0,3}}, {{0,1,3}}, {{2,2,0}},
    {{2,0,2}}, {{0,2,2}}, {{2,1,1}}, {{1,2,1}}, {{1,1,2}}
}};

// scalar-scalar
template <typename T>
bool gen_gs(
    T x,
    T y,
    T z,
    int32_t ider,
    std::vector<T>& last_point,          // Last position analyzed
    int32_t& last_der,                   // Last derivative analyzed
    const std::vector<int32_t>& types,   // Gaussian types     (nprims,)
    const std::vector<int32_t>& centers, // Gaussian centers   (nprims,
    const std::vector<T>& expons,        // Gaussian exponents (nprims,)
    const std::vector<T>& atpos,         // Atomic positions   (natoms, 3)
    T* gs, size_t nprims,               // Output: gs        (nprims,)
    T* gs1, size_t __r1, size_t __c1,   // Output: gs        (nprims, 3)
    T* gs2, size_t __r2, size_t __c2    // Output: gs        (nprims, 6)
);
