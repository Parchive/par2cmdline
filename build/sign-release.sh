#!/usr/bin/env bash

set -x

gpg_default_key=""
local_gpgkey="$(grep 'signingkey' .git/config | sed -e 's/.*=\s\+//')"

if [[ -n "$local_gpgkey" ]]; then
  echo "$local_gpgkey"
  gpg_default_key="--default-key $local_gpgkey"
fi

(
    cd ../par2release
    sha512sum *.tar* *.zip > checksums.sha512
    for file in *.tar* *.zip; do
        gpg $gpg_default_key --detach-sign $file
    done
)
