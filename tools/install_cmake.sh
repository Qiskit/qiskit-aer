#!/bin/bash

cmake --version
if [[ $? == 0 ]] ; then
    exit 0
fi

curl -L https://github.com/Kitware/CMake/releases/download/v3.16.4/cmake-3.16.4.tar.gz -o cmake-3.16.4.tar.gz
tar xzvf cmake-3.16.4.tar.gz
pushd cmake-3.16.4
./bootstrap
make
make install
popd
