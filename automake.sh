#!/bin/sh

set -e

aclocal
automake --warnings=all --add-missing
autoconf --warnings=all
