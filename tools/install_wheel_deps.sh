#!/bin/bash

curl -OL http://github.com/xianyi/OpenBLAS/archive/v0.2.20.tar.gz
tar xzf v0.2.20.tar.gz
pushd OpenBLAS-0.2.20
make
make PREFIX=/usr/local install
popd
