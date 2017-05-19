#!/bin/bash

export CFLAGS="-O3 -pipe -fstack-protector-strong"
export CXXFLAGS="-O3 -pipe -fstack-protector-strong"

# automake
./automake.sh
# configure
./configure
# make
make

