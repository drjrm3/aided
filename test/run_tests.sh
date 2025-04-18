#!/usr/bin/env bash

cd $(dirname ${0})

set -e

pip3 install .. #1> /dev/null 2> /dev/null

echo -n "[*] Version ... "
python3 -m aided --version

echo -n "[*] Linting ... "
pylint \
  --rcfile ../.pylintrc \
  --ignore version.py \
  ../aided

OMIT="__*__.py,version.py"
time coverage run -m \
  --source ../aided \
  --omit $OMIT \
  pytest -x -s -v -W ignore::DeprecationWarning unit_tests #-k TestGenChi


coverage report -m
