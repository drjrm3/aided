#!/usr/bin/env bash


if [[ 0 == 1 ]]; then
THIS_DIR=$(pwd)
pushd /home/drjrm3/code/aided/src > /dev/null
time python3 -m aided.core.edwfn -i ${THIS_DIR}/data/wfns/formamide/formamide.6311gss.b3lyp.wfn
time python3 -m aided.core.edwfns -i ${THIS_DIR}/tst.tmp
popd > /dev/null
exit 0
fi

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
time coverage run -m \
  --source ../src/aided \
  --omit $OMIT \
  pytest -s -v -W ignore::DeprecationWarning

coverage report -m
