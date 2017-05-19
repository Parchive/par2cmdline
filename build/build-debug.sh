#!/bin/bash

export CFLAGS="-g -O0"
export CXXFLAGS="-g -O0"

# automake
./automake.sh
# configure
./configure
# make
make

