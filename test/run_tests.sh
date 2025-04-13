#!/usr/bin/env bash

cd $(dirname ${0})

pip3 install ..

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
  pytest -x -s -v -W ignore::DeprecationWarning unit_tests

coverage report -m
