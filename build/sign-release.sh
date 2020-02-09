#!/usr/bin/env bash

(
    cd ../par2release
    sha512sum *.tar* *.zip > checksums.sha512
    for file in *.tar* *.zip; do
        gpg --detach-sign $file
    done
)

