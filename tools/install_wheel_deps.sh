#!/bin/bash

curl -OL http://github.com/xianyi/OpenBLAS/archive/v0.3.7.tar.gz
tar xzf v0.3.7.tar.gz
pushd OpenBLAS-0.3.7
make
make PREFIX=/usr/local install
popd
