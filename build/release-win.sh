#!/bin/bash

set -e

[[ -z $1 ]] && echo "you must give a version" && exit 1
version=$1

[[ ! -d ../par2release ]] && mkdir -p ../par2release

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
