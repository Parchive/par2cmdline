#!/usr/bin/env python3
"""Generate a fixture whose sets give their files a unicode filename packet.

A unicode filename packet gives a file's name in UTF-16, and a client uses it
in place of the name in the file description packet. par2 does not write the
packet, so each set is created as usual and the packet is added to every PAR2
file of it afterwards.

unicode.par2 names data_ascii.bin in its description packet and
data_é_\U0001f600.bin in its unicode filename packet, whose UTF-16 needs a
surrogate pair. bad.par2 names other_ascii.bin, and its unicode filename packet
holds a high surrogate with no low one, which is not UTF-16.
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
UNICODE_FILENAME = b"PAR 2.0\0UniFileN"

UNICODE_NAME = "data_é_\U0001f600.bin".encode("utf-16-le")
NOT_UTF16 = "other_".encode("utf-16-le") + b"\x00\xd8" + "x.bin".encode("utf-16-le")


def file_ids(path: pathlib.Path) -> tuple:
    """The set id, and the file id of each file description packet."""
    data = path.read_bytes()
    setid = None
    ids = []
    off = 0
    while off < len(data):
        if data[off:off + 8] != MAGIC:
            raise SystemExit(f"{path}: no packet magic at offset {off}")
        length = struct.unpack_from("<Q", data, off + 8)[0]
        setid = data[off + 32:off + 48]
        if data[off + 48:off + 64] == DESCRIPTION:
            ids.append(data[off + 64:off + 80])
        off += length
    return setid, ids


def packet(setid: bytes, fileid: bytes, name: bytes) -> bytes:
    """A unicode filename packet, its name padded with a zero unit to a
    multiple of 4 bytes."""
    body = fileid + name + b"\0" * (-len(name) % 4)
    length = 64 + len(body)
    # the packet hash covers everything from the set id on
    digest = hashlib.md5(setid + UNICODE_FILENAME + body).digest()
    return MAGIC + struct.pack("<Q", length) + digest + setid + UNICODE_FILENAME + body


def create(par2: pathlib.Path, temp: pathlib.Path, base: str, source: str, name: bytes, seed: int) -> None:
    # Each block holds different data, so a block cannot be matched against
    # another one.
    (temp / source).write_bytes(b"".join(b"%031d%032d\n" % (seed, index) for index in range(512))[:32768])
    subprocess.run(
        [str(par2), "c", "-q", "-s1024", "-c32", base + ".par2", source],
        cwd=temp,
        check=True,
    )
    for path in sorted(temp.glob(base + "*.par2")):
        setid, ids = file_ids(path)
        if not ids:
            raise SystemExit(f"{path}: no file description packet was found")
        with open(path, "ab") as out:
            for fileid in ids:
                out.write(packet(setid, fileid, name))


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: generate-unicode-name-fixture.py PAR2_BINARY OUTPUT_TAR_GZ"
        )

    par2 = pathlib.Path(sys.argv[1]).resolve()
    output = pathlib.Path(sys.argv[2]).resolve()

    with tempfile.TemporaryDirectory() as temp_name:
        temp = pathlib.Path(temp_name)

        create(par2, temp, "unicode", "data_ascii.bin", UNICODE_NAME, 1)
        create(par2, temp, "bad", "other_ascii.bin", NOT_UTF16, 2)

        with tarfile.open(output, "w:gz") as archive:
            for name in ["data_ascii.bin", "other_ascii.bin"] + sorted(
                os.path.basename(path) for path in glob.glob(str(temp / "*.par2"))
            ):
                archive.add(temp / name, arcname=name)


if __name__ == "__main__":
    main()
