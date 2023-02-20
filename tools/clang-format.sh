#!/bin/bash

cd `dirname $0`/..
IGNORE_FILES="src/misc/warnings.hpp"

ERROR=0
for FILE in `find src -iname "*.hpp" -or -iname "*.cpp"`; do

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

exit $ERROR
