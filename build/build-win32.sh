#!/bin/bash

export CFLAGS="-O3 -s -pipe -fstack-protector-strong"
export CXXFLAGS="-O3 -s -pipe -fstack-protector-strong"
export LDFLAGS="-static"

# automake
./automake.sh
# configure
./configure --host=i686-w64-mingw32
# make
make
