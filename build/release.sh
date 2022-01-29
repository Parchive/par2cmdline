#!/usr/bin/env bash

set -e

git clean -xfd
build/build.sh
make check
make distcheck
version=$(ls -1 par2cmdline*.tar.gz | tail -n1 | sed -e 's/par2cmdline-\([0-9\.]\+\)\.tar\.gz/\1/')
[[ -d ../par2release ]] && rm -r ../par2release
mkdir -p ../par2release
mv par2cmdline-$version.tar.gz ../par2release
(
    cd ../par2release
    zcat par2cmdline-$version.tar.gz | bzip2 > par2cmdline-$version.tar.bz2
)

# build/release-win.sh

git tag -a --sign v$version
