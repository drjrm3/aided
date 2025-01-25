#!/usr/bin/env bash

MIN_SCORE=9.0

cd $(dirname ${0})

pylint \
    --rcfile ../.pylintrc\
    --ignore version.py \
    ../src/connex > .pylint.tmp

PYLINT_SCORE=$(tail -n2 .pylint.tmp | cut -d'/' -f1 | awk 'NF{print $NF}')
cat .pylint.tmp

RC=`echo "$PYLINT_SCORE < $MIN_SCORE" | bc -l`

exit $RC
