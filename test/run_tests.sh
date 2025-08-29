#!/usr/bin/env bash

cd $(dirname ${0})

mypy ../aided --check-untyped-defs
echo $?

pylint \
  --rcfile ../.pylintrc \
  --ignore version.py \
  --fail-under 9 \
  ../aided

pylint \
  --rcfile .pylintrc \
  --ignore version.py \
  --fail-under 9 \
  --disable too-many-locals,too-few-public-methods,too-many-instance-attributes \
  ../test/unit_tests

set -e
echo -n "[*] Installing ... "
pip3 install .. 1> /dev/null 2> /dev/null
echo "done"
set +e

OMIT="__*__.py,version.py,unit_tests"
time coverage run -m \
  --source ../aided \
  --omit $OMIT \
  pytest -x -v -W ignore::DeprecationWarning \
  unit_tests # -s -k "wfn_dynamic_test"


coverage report -m
