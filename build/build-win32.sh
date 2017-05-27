#!/bin/bash

export CFLAGS="-O3 -pipe -fstack-protector-strong"
export CXXFLAGS="-O3 -pipe -fstack-protector-strong"
export LDFLAGS="-static"

# automake
./automake.sh
# configure
./configure --host=i686-w64-mingw32
# make
make
# strip
/usr/x86_64-w64-mingw32/bin/strip par2.exe
