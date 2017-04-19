#!/bin/sh

set -o errexit
set -o pipefail
set -o nounset

aclocal
automake --warnings=all --add-missing
autoconf --warnings=all
