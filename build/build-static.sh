#!/usr/bin/env bash

export CFLAGS="-O3 -pipe -fstack-protector-strong"
export CXXFLAGS="-O3 -pipe -fstack-protector-strong"
export LDFLAGS="-static -s"

# automake
./automake.sh
# configure
./configure
# make
make
