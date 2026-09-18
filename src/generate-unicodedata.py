#!/usr/bin/env python3
"""Write src/unicodedata.h from the Unicode database Python carries.

The tables are what is needed to compare two spellings of one name: the
canonical decomposition of each character, fully expanded so that a lookup
needs no recursion, and the canonical combining class of each mark, which
decides the order the marks are written in. Hangul is left out because its
decomposition is arithmetic.
"""

import unicodedata
from typing import List

HANGUL_FIRST = 0xAC00
HANGUL_LAST = 0xD7A3

LICENCE = """\
//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2026 Michael Nightingale
//
//  par2cmdline is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  par2cmdline is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
"""


def expand(codepoint: int) -> List[int]:
    decomposition = unicodedata.decomposition(chr(codepoint))
    if not decomposition or decomposition.startswith("<"):
        return [codepoint]
    expanded: List[int] = []
    for part in decomposition.split():
        expanded.extend(expand(int(part, 16)))
    return expanded


def main() -> None:
    mappings = []
    for codepoint in range(0x110000):
        if HANGUL_FIRST <= codepoint <= HANGUL_LAST:
            continue
        decomposition = unicodedata.decomposition(chr(codepoint))
        if decomposition and not decomposition.startswith("<"):
            mappings.append((codepoint, expand(codepoint)))

    classes = [
        (codepoint, unicodedata.combining(chr(codepoint)))
        for codepoint in range(0x110000)
        if unicodedata.combining(chr(codepoint))
    ]

    width = max(len(mapping) for _, mapping in mappings)

    with open("src/unicodedata.h", "w") as out:
        out.write(LICENCE)
        out.write(
            f"""
// Written by src/generate-unicodedata.py from Unicode {unicodedata.unidata_version}.
// Do not edit.

#ifndef __UNICODEDATA_H__
#define __UNICODEDATA_H__

namespace par2
{{

// The characters one character is written as when it is written out in full,
// padded with zeros. Sorted by codepoint.
struct CanonicalMapping
{{
  u32 codepoint;
  u32 mapping[{width}];
}};

static const CanonicalMapping canonicalmappings[] =
{{
"""
        )
        for codepoint, mapping in mappings:
            padded = mapping + [0] * (width - len(mapping))
            written = ", ".join(f"0x{value:04X}" for value in padded)
            out.write(f"  {{ 0x{codepoint:04X}, {{ {written} }} }},\n")
        out.write(
            """};

// The order marks are written in. A character which is not here is a starter,
// whose class is zero. Sorted by codepoint.
struct CombiningClass
{
  u32 codepoint;
  u8 combiningclass;
};

static const CombiningClass combiningclasses[] =
{
"""
        )
        for codepoint, combining in classes:
            out.write(f"  {{ 0x{codepoint:04X}, {combining} }},\n")
        out.write(
            """};

} // namespace par2

#endif // __UNICODEDATA_H__
"""
        )

    print(f"{len(mappings)} mappings, {len(classes)} combining classes, width {width}")


if __name__ == "__main__":
    main()
