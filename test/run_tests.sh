#!/usr/bin/env bash

cd $(dirname ${0})

export PYTHONPATH=../src:$PYTHONPATH

echo -n "[*] Version ... "
python3 -m aided --version

echo -n "[*] Linting ... "
pylint \
  --rcfile ../.pylintrc \
  --ignore version.py \
  ../src/aided

OMIT="__*__.py,version.py"
coverage run -m \
  --source ../src/aided \
  --omit $OMIT \
  pytest -v -x -W ignore::DeprecationWarning

coverage report -m
