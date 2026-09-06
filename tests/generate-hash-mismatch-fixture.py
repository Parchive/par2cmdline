#!/usr/bin/env python3
"""Generate a hash mismatch fixture from a freshly created PAR2 file.

Every block of the data file still matches its verification entry, but one of
the hashes in the file description packets does not match the file. Breaking
"16k" is noticed by an ordinary verify, "full" only with
--full-hash.
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

# Past the 64 byte header and the file id come the two hashes
HASH_OFFSET = {"full": 64 + 16, "16k": 64 + 16 + 16}


def break_hash(path, field):
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
            data[off + HASH_OFFSET[field]] ^= 0xFF
            # the packet carries its own hash, over everything from the set id on
            digest = hashlib.md5(bytes(data[off + 32:off + length])).digest()
            data[off + 16:off + 32] = digest
            altered += 1

        off += length

    path.write_bytes(data)
    return altered


def main():
    if len(sys.argv) != 4 or sys.argv[3] not in HASH_OFFSET:
        raise SystemExit(
            "usage: generate-hash-mismatch-fixture.py PAR2_BINARY OUTPUT_TAR_GZ full|16k"
        )

    par2 = pathlib.Path(sys.argv[1]).resolve()
    output = pathlib.Path(sys.argv[2]).resolve()
    field = sys.argv[3]

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
            altered += break_hash(pathlib.Path(name), field)

        if altered == 0:
            raise SystemExit("no file description packets were found")

        with tarfile.open(output, "w:gz") as archive:
            archive.add(data, arcname="data.bin")
            for name in sorted(glob.glob(str(temp / "*.par2"))):
                archive.add(name, arcname=os.path.basename(name))


if __name__ == "__main__":
    main()
