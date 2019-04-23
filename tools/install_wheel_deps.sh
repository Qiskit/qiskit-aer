#!/bin/bash

curl -OL http://github.com/xianyi/OpenBLAS/archive/v0.2.20.tar.gz
tar xzvf v0.2.20.tar.gz
cd OpenBLAS-0.2.20
make
make install
