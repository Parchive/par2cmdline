// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "par2cmdline",
    products: [
        .library(
            name: "par2cmdline",
            targets: [
                "par2cmdline",
            ],
        ),
    ],
    targets: [
        .target(
            name: "par2cmdline",
            path: "src",
            sources: [
                "commandline.cpp",
                "crc.cpp",
                "creatorpacket.cpp",
                "criticalpacket.cpp",
                "datablock.cpp",
                "descriptionpacket.cpp",
                "diskfile.cpp",
                "filechecksummer.cpp",
                "galois.cpp",
                "libpar2.cpp",
                "mainpacket.cpp",
                "md5.cpp",
                "par1fileformat.cpp",
                "par1repairer.cpp",
                "par1repairersourcefile.cpp",
                "par2creator.cpp",
                "par2creatorsourcefile.cpp",
                "par2fileformat.cpp",
                "par2repairer.cpp",
                "par2repairersourcefile.cpp",
                "recoverypacket.cpp",
                "reedsolomon.cpp",
                "utf8.cpp",
                "verificationhashtable.cpp",
                "verificationpacket.cpp",
            ],
            publicHeadersPath: "",
            cxxSettings: [
                .define("NDEBUG"),
                .define("PACKAGE", to: "\"par2cmdline\""),
                .define("VERSION", to: "\"1.0.0\""),
            ],
        ),
    ],
    cxxLanguageStandard: .cxx14,
)
