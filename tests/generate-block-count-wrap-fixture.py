#!/usr/bin/env python3
"""Generate block-count-wrap.tar.gz from a freshly created PAR2 file."""

import hashlib
import pathlib
import struct
import subprocess
import sys
import tarfile
import tempfile


def rewrite_fixture(path):
    data = bytearray(path.read_bytes())

    magic = b"PAR2\0PKT"
    main_type = b"PAR 2.0\0Main\0\0\0\0"
    desc_type = b"PAR 2.0\0FileDesc"
    main = None
    descs = []
    off = 0

    while True:
        idx = data.find(magic, off)
        if idx < 0:
            break
        length = struct.unpack_from("<Q", data, idx + 8)[0]
        ptype = bytes(data[idx + 48:idx + 64])
        if ptype == main_type:
            main = (idx, length)
        elif ptype == desc_type:
            descs.append((idx, length))
        off = idx + max(length, 1)

    if main is None or len(descs) < 2:
        raise SystemExit("could not find expected PAR2 packets")

    sizes = [0x80000000 * 4, 0x80000001 * 4]
    fileids = []

    for (idx, length), size in zip(descs[:2], sizes):
        length_offset = idx + 64 + 16 + 16 + 16
        struct.pack_into("<Q", data, length_offset, size)

        name_start = idx + 64 + 16 + 16 + 16 + 8
        name = bytes(data[name_start:idx + length]).split(b"\0", 1)[0]
        hash16k = bytes(data[idx + 64 + 16 + 16:idx + 64 + 16 + 16 + 16])
        fileid = hashlib.md5(hash16k + struct.pack("<Q", size) + name).digest()
        data[idx + 64:idx + 80] = fileid
        fileids.append(fileid)

    main_idx, main_length = main
    for index, fileid in enumerate(fileids):
        start = main_idx + 64 + 8 + 4 + index * 16
        data[start:start + 16] = fileid

    setid = hashlib.md5(data[main_idx + 64:main_idx + main_length]).digest()
    for idx, _length in [main] + descs[:2]:
        data[idx + 32:idx + 48] = setid
    for idx, length in [main] + descs[:2]:
        data[idx + 16:idx + 32] = hashlib.md5(data[idx + 32:idx + length]).digest()

    path.write_bytes(data)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: generate-block-count-wrap-fixture.py PAR2_BINARY OUTPUT_TAR_GZ"
        )

    par2 = pathlib.Path(sys.argv[1]).resolve()
    output = pathlib.Path(sys.argv[2]).resolve()

    with tempfile.TemporaryDirectory() as temp_name:
        temp = pathlib.Path(temp_name)
        (temp / "a.bin").write_bytes(b"aaaa")
        (temp / "b.bin").write_bytes(b"bbbb")
        recovery = temp / "recovery.par2"

        subprocess.run(
            [str(par2), "c", "-q", "-s4", "-r1", str(recovery), "a.bin", "b.bin"],
            cwd=temp,
            check=True,
        )

        rewrite_fixture(recovery)
        with tarfile.open(output, "w:gz") as archive:
            archive.add(recovery, arcname="recovery.par2")


if __name__ == "__main__":
    main()
