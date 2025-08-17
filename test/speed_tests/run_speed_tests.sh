#!/usr/bin/env bash

set -e

cd $(dirname ${0})

### _gen_chi without _primitives.gpow:
#_gen_chi runs at a speed of  13.4K calls/sec for ider=0.
#_gen_chi runs at a speed of   8.3K calls/sec for ider=1.
#_gen_chi runs at a speed of   5.2K calls/sec for ider=2.

### _gen_chi with _primitives.gpow:
#_gen_chi runs at a speed of  15.1K calls/sec for ider=0.
#_gen_chi runs at a speed of   9.2K calls/sec for ider=1.
#_gen_chi runs at a speed of   5.6K calls/sec for ider=2.

### 2025.04.13 - Update to gen_gs with C++
#_gen_chi runs at a speed of  55.7K calls/sec for ider=0.
#_gen_chi runs at a speed of  48.8K calls/sec for ider=1.
#_gen_chi runs at a speed of  40.7K calls/sec for ider=2.

### 2025.08.26 - Speeds reduced with new implementation of gen_gs somehow.
#_gen_gs runs at a speed of  45.8K calls/sec for ider=0.
#_gen_gs runs at a speed of  44.6K calls/sec for ider=1.
#_gen_gs runs at a speed of  42.5K calls/sec for ider=2.
#gpow(x, 0) ..... runs at a speed of 378.2K evals/sec.
#gpow(xs, 0) .... runs at a speed of  21.1M evals/sec.
#gpow(0, n) ..... runs at a speed of 392.9K evals/sec.
#gpow(0, ns) .... runs at a speed of  18.2M evals/sec.
#gpow(x, n) ..... runs at a speed of   2.0M evals/sec.
#gpow(xs, ns) ... runs at a speed of   9.6M evals/sec.


pip3 install ../.. 1> install.out.tmp 2> install.err.tmp

python3 -m core \
    --test edwfn \
    --wfnfile ../data/wfns/formamide/formamide.6311gss.b3lyp.wfn \
    --num_iters 50000

python3 -m core \
    --test math
