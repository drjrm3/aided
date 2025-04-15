#!/usr/bin/env python3
"""
Speed tests for mathematical functions in aided.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from datetime import datetime

import numpy as np


#from aided.math.primitives import gpow
# When using aided.math.primitives the speed is:
# gpow(x, 0) ..... runs at a speed of 138.6K evals/sec.
# gpow(xs, 0) .... runs at a speed of   7.0M evals/sec.
# gpow(0, n) ..... runs at a speed of 151.5K evals/sec.
# gpow(0, ns) .... runs at a speed of   7.4M evals/sec.
# gpow(x, n) ..... runs at a speed of   1.2M evals/sec.
# gpow(xs, ns) ... runs at a speed of   9.4M evals/sec.

from aided.math._primitives import gpow
# When using aided.math.primitives_cpp the speed is:
# gpow(x, 0) ..... runs at a speed of 359.6K evals/sec.
# gpow(xs, 0) .... runs at a speed of  17.5M evals/sec.
# gpow(0, n) ..... runs at a speed of 384.5K evals/sec.
# gpow(0, ns) .... runs at a speed of  16.2M evals/sec.
# gpow(x, n) ..... runs at a speed of   1.9M evals/sec.
# gpow(xs, ns) ... runs at a speed of   8.2M evals/sec.



def test_gpow(num_iters: int, nvals: int = 100):
    """Test the gpow function."""
    xs = -100 + 200 * np.random.random(num_iters)
    xss = -100 + 200 * np.random.random((num_iters, nvals))
    ns = np.random.randint(-10, 20, num_iters)
    nss = np.random.randint(-10, 20, (num_iters, nvals))

    # Ensure ns does not have a value of 0 in it.
    while 0 in ns:
        mask = ns == 0
        ns[mask] = np.random.randint(-10, 20, num_iters)[mask]

    ##############################################
    # Ensure that anything raised to the 0 is 1. #
    ##############################################
    t1 = datetime.now()
    for x in xs:
        val = gpow(x, 0)
    t2 = datetime.now()
    tdiff = (t2 - t1).total_seconds()
    rate = (num_iters / tdiff) / 1e+3
    print(f"gpow(x, 0) ..... runs at a speed of {rate:5.1f}K evals/sec.")

    ##############################################
    ### Test gpow with a 2D array of x values. ###
    ##############################################
    t1 = datetime.now()
    for xs in xss:
        vals = gpow(xs, 0)
    t2 = datetime.now()
    tdiff = (t2 - t1).total_seconds()
    rate = ((num_iters * nvals) / tdiff) / 1e+6
    print(f"gpow(xs, 0) .... runs at a speed of {rate:5.1f}M evals/sec.")

    ###########################
    # 0 to any exponent is 0. #
    ###########################
    t1 = datetime.now()
    for n in ns:
        val = gpow(0, n)
    t2 = datetime.now()
    tdiff = (t2 - t1).total_seconds()
    rate = (num_iters / tdiff) / 1e+3
    print(f"gpow(0, n) ..... runs at a speed of {rate:5.1f}K evals/sec.")

    ##########################################
    # Test gpow with a 2D array of n values. #
    ##########################################
    t1 = datetime.now()
    for ns in nss:
        vals = gpow(0, ns)
    t2 = datetime.now()
    tdiff = (t2 - t1).total_seconds()
    rate = ((num_iters * nvals) / tdiff) / 1e+6
    print(f"gpow(0, ns) .... runs at a speed of {rate:5.1f}M evals/sec.")

    #####################
    # Arbitrary x and n #
    #####################
    t1 = datetime.now()
    for x, n in zip(xs, ns):
        val = gpow(x, n)
    t2 = datetime.now()
    tdiff = (t2 - t1).total_seconds()
    rate = (num_iters / tdiff) / 1e+6
    print(f"gpow(x, n) ..... runs at a speed of {rate:5.1f}M evals/sec.")

    ##########################################
    # Test with 2D array of x and n values.a #
    ##########################################
    t1 = datetime.now()
    for xs, ns in zip(xss, nss):
        vals = gpow(xs, ns)
    t2 = datetime.now()
    tdiff = (t2 - t1).total_seconds()
    rate = ((num_iters * nvals) / tdiff) / 1e+6
    print(f"gpow(xs, ns) ... runs at a speed of {rate:5.1f}M evals/sec.")

def main(args):
    """Main routine for speed tests."""

    test_gpow(args.num_iters)


