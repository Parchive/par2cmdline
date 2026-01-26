##  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
##  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
##
##  Copyright (c) 2003 Peter Brian Clements
##  Copyright (c) 2019 Michael D. Nahas
##
##  par2cmdline is free software; you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation; either version 2 of the License, or
##  (at your option) any later version.
##
##  par2cmdline is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program; if not, write to the Free Software
##  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

AUTOMAKE_OPTIONS = subdir-objects

.DELETE_ON_ERROR:

noinst_LIBRARIES = libpar2.a
libpar2_a_SOURCES = src/crc.cpp src/crc.h \
	src/creatorpacket.cpp src/creatorpacket.h \
	src/criticalpacket.cpp src/criticalpacket.h \
	src/datablock.cpp src/datablock.h \
	src/descriptionpacket.cpp src/descriptionpacket.h \
	src/diskfile.cpp src/diskfile.h \
	src/filechecksummer.cpp src/filechecksummer.h \
	src/galois.cpp src/galois.h \
	src/letype.h \
	src/mainpacket.cpp src/mainpacket.h \
	src/md5.cpp src/md5.h \
	src/par1fileformat.cpp src/par1fileformat.h \
	src/par1repairer.cpp src/par1repairer.h \
	src/par1repairersourcefile.cpp src/par1repairersourcefile.h \
	src/par2creator.cpp src/par2creator.h \
	src/par2creatorsourcefile.cpp src/par2creatorsourcefile.h \
	src/par2fileformat.cpp src/par2fileformat.h \
	src/par2repairer.cpp src/par2repairer.h \
	src/par2repairersourcefile.cpp src/par2repairersourcefile.h \
	src/recoverypacket.cpp src/recoverypacket.h \
	src/reedsolomon.cpp src/reedsolomon.h \
	src/verificationhashtable.cpp src/verificationhashtable.h \
	src/verificationpacket.cpp src/verificationpacket.h \
	src/libpar2.cpp src/libpar2.h src/libpar2internal.h \
	src/utf8.cpp src/utf8.h


bin_PROGRAMS = par2
man_MANS = man/par2.1

par2_SOURCES = src/par2cmdline.cpp \
	src/commandline.cpp src/commandline.h
par2_LDADD = libpar2.a

LDADD = -lstdc++
AM_CXXFLAGS = -Wall -std=c++14 $(OPENMP_CXXFLAGS)

EXTRA_DIST = PORTING ROADMAP \
			 man/par2.1 \
			 automake.sh \
			 tests/flatdata.tar.gz \
			 tests/flatdata-par1files.tar.gz \
			 tests/flatdata-par2files.tar.gz \
			 tests/subdirdata.tar.gz \
			 tests/subdirdata-par2files-unix.tar.gz \
			 tests/subdirdata-par2files-win.tar.gz \
			 tests/smallsubdirdata.tar.gz \
			 tests/smallsubdirdata-par2files.tar.gz \
			 tests/100blocks.tar.gz \
			 tests/readbeyondeof.tar.gz \
			 tests/par2-0.6.8-crash.tar.gz \
			 tests/bug44.tar.gz \
			 tests/bug128-parfiles.tar.gz \
			 tests/bug190.tar.gz \
			 tests/testfuncs.ps1 \
			 tests/build_unit_tests.ps1 \
			 tests/run_tests.ps1 \
			 tests/test1 \
			 tests/test1.ps1 \
			 tests/test2 \
			 tests/test2.ps1 \
			 tests/test3 \
			 tests/test3.ps1 \
			 tests/test4 \
			 tests/test4.ps1 \
			 tests/test5 \
			 tests/test5.ps1 \
			 tests/test5rk \
			 tests/test5rk.ps1 \
			 tests/test6 \
			 tests/test6.ps1 \
			 tests/test7 \
			 tests/test7.ps1 \
			 tests/test8 \
			 tests/test8.ps1 \
			 tests/test9 \
			 tests/test9.ps1 \
			 tests/test10 \
			 tests/test10.ps1 \
			 tests/test11 \
			 tests/test11.ps1 \
			 tests/test12 \
			 tests/test12.ps1 \
			 tests/test13 \
			 tests/test13.ps1 \
			 tests/test14 \
			 tests/test14.ps1 \
			 tests/test15 \
			 tests/test15.ps1 \
			 tests/test16 \
			 tests/test16.ps1 \
			 tests/test17 \
			 tests/test17.ps1 \
			 tests/test18 \
			 tests/test18.ps1 \
			 tests/test19 \
			 tests/test19.ps1 \
			 tests/test20 \
			 tests/test20.ps1 \
			 tests/test21 \
			 tests/test21.ps1 \
			 tests/test22 \
			 tests/test22.ps1 \
			 tests/test23 \
			 tests/test23.ps1 \
			 tests/test24 \
			 tests/test24.ps1 \
			 tests/test25 \
			 tests/test25.ps1 \
			 tests/test26 \
			 tests/test26.ps1 \
			 tests/test27 \
			 tests/test27.ps1 \
			 tests/test28 \
			 tests/test28.ps1 \
			 tests/test29 \
			 tests/test29.ps1 \
			 tests/test30 \
			 tests/test30.ps1 \
			 tests/test31 \
			 tests/test31.ps1 \
			 tests/test32 \
			 tests/test32.ps1 \
			 tests/test33 \
			 tests/test33.ps1 \
			 tests/test34 \
			 tests/test34.ps1 \
			 tests/test35 \
			 tests/test35.ps1 \
			 tests/unit_tests \
			 tests/unit_tests.ps1


# Programs that need to be compiled for the test suite.
# These are the unit tests.
check_PROGRAMS = tests/letype_test tests/crc_test tests/md5_test tests/diskfile_test tests/libpar2_test tests/commandline_test tests/descriptionpacket_test tests/criticalpacket_test tests/reedsolomon_test tests/galois_test tests/utf8_test

tests_letype_test_SOURCES = src/letype_test.cpp src/letype.h

tests_crc_test_SOURCES = src/crc_test.cpp src/crc.cpp src/crc.h

tests_md5_test_SOURCES = src/md5_test.cpp src/md5.cpp src/md5.h

tests_diskfile_test_SOURCES = src/diskfile_test.cpp src/diskfile.cpp src/diskfile.h
tests_diskfile_test_LDADD = libpar2.a

tests_libpar2_test_SOURCES = src/libpar2_test.cpp src/libpar2.h
tests_libpar2_test_LDADD = libpar2.a

tests_commandline_test_SOURCES = src/commandline_test.cpp src/commandline.cpp src/commandline.h
tests_commandline_test_LDADD = libpar2.a

tests_descriptionpacket_test_SOURCES = src/descriptionpacket_test.cpp src/descriptionpacket.cpp src/descriptionpacket.h
tests_descriptionpacket_test_LDADD = libpar2.a

tests_criticalpacket_test_SOURCES = src/criticalpacket_test.cpp src/criticalpacket.cpp src/criticalpacket.h
tests_criticalpacket_test_LDADD = libpar2.a

tests_reedsolomon_test_SOURCES = src/reedsolomon_test.cpp src/reedsolomon.cpp src/reedsolomon.h

tests_galois_test_SOURCES = src/galois_test.cpp src/galois.cpp src/galois.h

tests_utf8_test_SOURCES = src/utf8_test.cpp src/utf8.cpp src/utf8.h


# List of all tests.
# tests/test* are integration tests that use the binary.
# $(check_PROGRAMS) is the list of compiled unit tests.
TESTS = tests/test1 \
		tests/test2 \
		tests/test3 \
		tests/test4 \
		tests/test5 \
		tests/test5rk \
		tests/test6 \
		tests/test7 \
		tests/test8 \
		tests/test9 \
		tests/test10 \
		tests/test11 \
		tests/test12 \
		tests/test13 \
		tests/test14 \
		tests/test15 \
		tests/test16 \
		tests/test17 \
		tests/test18 \
		tests/test19 \
		tests/test20 \
		tests/test21 \
		tests/test22 \
		tests/test23 \
		tests/test24 \
		tests/test25 \
		tests/test26 \
		tests/test27 \
		tests/test28 \
		tests/test29 \
		tests/test30 \
		tests/test31 \
		tests/test32 \
		tests/test33 \
		tests/test34 \
		tests/test35 \
		tests/utf8_test \
		tests/unit_tests

install-exec-hook :
	cd $(DESTDIR)$(bindir)/ && \
	ln -sf par2$(EXEEXT) par2create$(EXEEXT) && \
	ln -sf par2$(EXEEXT) par2verify$(EXEEXT) && \
	ln -sf par2$(EXEEXT) par2repair$(EXEEXT)

install-data-hook :
	cd $(DESTDIR)$(man1dir)/ && \
	ln -sf par2.1 par2create.1 && \
	ln -sf par2.1 par2verify.1 && \
	ln -sf par2.1 par2repair.1

uninstall-hook :
	rm -f $(DESTDIR)$(bindir)/par2create$(EXEEXT)
	rm -f $(DESTDIR)$(bindir)/par2verify$(EXEEXT)
	rm -f $(DESTDIR)$(bindir)/par2repair$(EXEEXT)
	rm -f $(DESTDIR)$(man1dir)/par2create.1
	rm -f $(DESTDIR)$(man1dir)/par2verify.1
	rm -f $(DESTDIR)$(man1dir)/par2repair.1
