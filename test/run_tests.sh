#!/usr/bin/env bash

cd $(dirname ${0})

set -e
echo -n "[*] Installing ... "
pip3 install .. 1> /dev/null 2> /dev/null
echo "done"
set +e

#pylint \
#  --rcfile ../.pylintrc \
#  --ignore version.py \
#  ../aided

OMIT="__*__.py,version.py"
time coverage run -m \
  --source ../aided \
  --omit $OMIT \
  pytest -x -v -W ignore::DeprecationWarning unit_tests #-k Gaussian


coverage report -m
