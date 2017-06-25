#!/bin/bash

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
git clean -xfd
build/build-win32.sh
mv par2.exe ../par2release
(
    cd ../par2release
    zip par2cmdline-$version-win-x86.zip par2.exe
    rm par2.exe
)
git clean -xfd
build/build-win64.sh
mv par2.exe ../par2release
(
    cd ../par2release
    zip par2cmdline-$version-win-x64.zip par2.exe
    rm par2.exe
)

(
    cd ../par2release
    sha512sum *.tar* *.zip > checksums.sha512
    for file in *.tar* *.zip; do
        gpg --detach-sign $file
    done
)

git tag -a --sign v$version
