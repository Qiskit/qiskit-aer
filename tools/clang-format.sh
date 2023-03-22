#!/bin/bash

cd `dirname $0`/..
TARGET_DIRS="src qiskit_aer contrib test"
IGNORE_FILES="src/misc/warnings.hpp"

ERROR=0
for DIR in $TARGET_DIRS; do
  for FILE in `find $DIR -iname "*.hpp" -or -iname "*.cpp"`; do

    IGNORE=false
    for IGNORE_FILE in $IGNORE_FILES; do
      if [ "$IGNORE_FILE" = "$FILE" ]; then
        IGNORE=true
        break
      fi
    done

    if $IGNORE; then continue; fi

    clang-format -style=file $* $FILE
    if [ "$?" != "0" ]; then
      ERROR=1
    fi
  done
done

exit $ERROR
