#!/usr/bin/env bash

set -e

cmake -B build-cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-pipe -fstack-protector-strong" \
  -DCMAKE_CXX_FLAGS="-pipe -fstack-protector-strong"
cmake --build build-cmake -j
