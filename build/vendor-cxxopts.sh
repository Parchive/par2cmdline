#!/usr/bin/env bash

# Copies cxxopts into libs/cxxopts at the release given, or at the one below.
# The header records its own version.

set -e

version=${1:-3.3.1}
url=https://raw.githubusercontent.com/jarro2783/cxxopts/v$version

cd "$(dirname "$0")/.."
curl -fsSL "$url/include/cxxopts.hpp" -o libs/cxxopts/cxxopts.hpp
curl -fsSL "$url/LICENSE" -o libs/cxxopts/LICENSE
