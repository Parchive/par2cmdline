#!/usr/bin/env bash

set -e

cmake -B build-cmake -DCMAKE_BUILD_TYPE=Debug
cmake --build build-cmake -j
