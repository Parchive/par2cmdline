#!/usr/bin/env python3
"""Generate full-hash-mismatch.tar.gz from a freshly created PAR2 file.

Every block of the data file still matches its verification entry, but the
whole file hash recorded in the file description packets does not match the
file. Verification only notices with --force-full-hash-verify.
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


def break_whole_file_hash(path):
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
            # header 64, then fileid 16, then the hash of the whole file
            data[off + 80] ^= 0xFF
            # the packet carries its own hash, over everything from the set id on
            digest = hashlib.md5(bytes(data[off + 32:off + length])).digest()
            data[off + 16:off + 32] = digest
            altered += 1

        off += length

    path.write_bytes(data)
    return altered


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: generate-full-hash-mismatch-fixture.py PAR2_BINARY OUTPUT_TAR_GZ"
        )

    par2 = pathlib.Path(sys.argv[1]).resolve()
    output = pathlib.Path(sys.argv[2]).resolve()

    with tempfile.TemporaryDirectory() as temp_name:
        temp = pathlib.Path(temp_name)
        data = temp / "data.bin"
        data.write_bytes(bytes(range(256)) * 128)

        subprocess.run(
            [str(par2), "c", "-q", "-s1024", "-c4", "recovery.par2", "data.bin"],
            cwd=temp,
            check=True,
        )

        altered = 0
        for name in sorted(glob.glob(str(temp / "*.par2"))):
            altered += break_whole_file_hash(pathlib.Path(name))

        if altered == 0:
            raise SystemExit("no file description packets were found")

        with tarfile.open(output, "w:gz") as archive:
            archive.add(data, arcname="data.bin")
            for name in sorted(glob.glob(str(temp / "*.par2"))):
                archive.add(name, arcname=os.path.basename(name))


if __name__ == "__main__":
    main()
