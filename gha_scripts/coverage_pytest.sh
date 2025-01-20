#!/usr/bin/env bash

MIN_SCORE=90

# CD into test directory.
cd ./test

export PYTHONPATH=../src:$PYTHONPATH

OMIT="__*__.py,version.py,cli.py"
coverage run -m \
    --source ../src/aided \
    --omit $OMIT \
    pytest --tb=short -W ignore::DeprecationWarning
RC=$?
if [[ $RC != 0 ]]; then
    exit $RC
fi

coverage report -m > .coverage.out

cat .coverage.out

COVERAGE_SCORE=`tail .coverage.out | grep 'TOTAL' | awk '{print $NF}' | sed 's/%//g'`

RC=`echo "$COVERAGE_SCORE < $MIN_SCORE" | bc -l`
