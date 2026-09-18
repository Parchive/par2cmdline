#!/usr/bin/env python3
"""Generate a fixture whose recorded name is not UTF-8.

A PAR2 set records a name as bytes and says nothing about what they were
written in, so a set written on a machine using a code page records the name
in that code page. Here the file on disk is named in UTF-8 and the set records
the same name in Windows-1252, which is what a Windows client of that age
wrote.

The name is replaced in the packet rather than the set being created from a
file of that name, because a file system which insists on UTF-8 will not hold
one. It is one byte shorter, so the byte it leaves behind is one of the zeros
the name is padded with and the packet keeps its length. The file id is not
recomputed: it identifies the file rather than describing it, and par2 reads
it as it is recorded.
"""

import glob
import hashlib
import os
import pathlib
import struct
import subprocess
import sys
import tarfile
import tempfile

MAGIC = b"PAR2\0PKT"
DESCRIPTION = b"PAR 2.0\0FileDesc"

UTF8_NAME = "data_è.bin"
CODEPAGE_NAME = UTF8_NAME.encode("windows-1252")


def rewrite_name(path: pathlib.Path) -> int:
    data = bytearray(path.read_bytes())
    off = 0
    altered = 0

    while off < len(data):
        if data[off:off + 8] != MAGIC:
            raise SystemExit(f"{path}: no packet magic at offset {off}")
        length = struct.unpack_from("<Q", data, off + 8)[0]
        if length < 64 or off + length > len(data):
            raise SystemExit(f"{path}: bad packet length at offset {off}")

        if data[off + 48:off + 64] == DESCRIPTION:
            # Past the header come the file id and the three other fields the
            # packet records before the name.
            name = off + 64 + 16 + 16 + 16 + 8
            recorded = UTF8_NAME.encode("utf-8")

            if data[name:name + len(recorded)] != recorded:
                raise SystemExit(f"{path}: the packet does not record {UTF8_NAME}")

            data[name:name + len(recorded)] = CODEPAGE_NAME + b"\0"

            # the packet carries its own hash, over everything from the set id on
            digest = hashlib.md5(bytes(data[off + 32:off + length])).digest()
            data[off + 16:off + 32] = digest
            altered += 1

        off += length

    path.write_bytes(data)
    return altered


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: generate-codepage-name-fixture.py PAR2_BINARY OUTPUT_TAR_GZ"
        )

    par2 = pathlib.Path(sys.argv[1]).resolve()
    output = pathlib.Path(sys.argv[2]).resolve()

    with tempfile.TemporaryDirectory() as temp_name:
        temp = pathlib.Path(temp_name)
        data = temp / UTF8_NAME
        # Each block holds different data, so a block cannot be matched
        # against another one.
        data.write_bytes(b"".join(b"%063d\n" % index for index in range(512)))

        subprocess.run(
            [str(par2), "c", "-q", "-s1024", "-c32", "recovery.par2", UTF8_NAME],
            cwd=temp,
            check=True,
        )

        altered = 0
        for name in sorted(glob.glob(str(temp / "*.par2"))):
            altered += rewrite_name(pathlib.Path(name))

        if altered == 0:
            raise SystemExit("no file description packets were found")

        with tarfile.open(output, "w:gz") as archive:
            archive.add(data, arcname=UTF8_NAME)
            for name in sorted(glob.glob(str(temp / "*.par2"))):
                archive.add(name, arcname=os.path.basename(name))


if __name__ == "__main__":
    main()
