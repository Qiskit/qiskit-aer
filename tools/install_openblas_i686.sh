#!/bin/sh

wget https://github.com/xianyi/OpenBLAS/releases/download/v0.3.21/OpenBLAS-0.3.21.zip
unzip OpenBLAS-0.3.21.zip
cd OpenBLAS-0.3.21
make TARGET=KATMAI
make TARGET=KATMAI PREFIX=/usr/ install
