#!/usr/bin/env bash

cd $(dirname ${0})

### _gen_chi without _primitives.gpow:
#_gen_chi runs at a speed of  13.4K calls/sec for ider=0.
#_gen_chi runs at a speed of   8.3K calls/sec for ider=1.
#_gen_chi runs at a speed of   5.2K calls/sec for ider=2.

### _gen_chi with _primitives.gpow:
#_gen_chi runs at a speed of  15.1K calls/sec for ider=0.
#_gen_chi runs at a speed of   9.2K calls/sec for ider=1.
#_gen_chi runs at a speed of   5.6K calls/sec for ider=2.

pip3 install ../..

python3 -m core \
    --test edwfn \
    --wfnfile ../data/wfns/formamide/formamide.6311gss.b3lyp.wfn

python3 -m core \
    --test math
