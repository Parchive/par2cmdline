#!/usr/bin/env bash

set -e

git clean -xfd
build/build.sh
ctest --test-dir build-cmake --output-on-failure
version=$(sed -n 's/^  VERSION \([0-9.]*\)$/\1/p' CMakeLists.txt)
[[ -d ../par2release ]] && rm -r ../par2release
mkdir -p ../par2release
git archive --prefix=par2cmdline-$version/ -o ../par2release/par2cmdline-$version.tar.gz HEAD
(
    cd ../par2release
    zcat par2cmdline-$version.tar.gz | bzip2 > par2cmdline-$version.tar.bz2
)

# build/release-win.sh

git tag -a --sign v$version
