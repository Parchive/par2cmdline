##  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
##  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
##
##  Copyright (c) 2003 Peter Brian Clements
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

bin_PROGRAMS = par2
man_MANS = man/par2.1

par2_SOURCES = src/par2cmdline.cpp src/par2cmdline.h \
	src/commandline.cpp src/commandline.h \
	src/crc.cpp src/crc.h \
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
	src/verificationpacket.cpp src/verificationpacket.h

LDADD = -lstdc++
AM_CXXFLAGS = -Wall $(OPENMP_CXXFLAGS)

EXTRA_DIST = PORTING ROADMAP par2cmdline.sln par2cmdline.vcproj \
			 man/par2.1 \
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
			 tests/test1 \
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
			 tests/test24

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
		tests/test24

install-exec-hook :
	ln -sf $(bindir)/par2$(EXEEXT) $(DESTDIR)$(bindir)/par2create$(EXEEXT)
	ln -sf $(bindir)/par2$(EXEEXT) $(DESTDIR)$(bindir)/par2verify$(EXEEXT)
	ln -sf $(bindir)/par2$(EXEEXT) $(DESTDIR)$(bindir)/par2repair$(EXEEXT)

uninstall-hook :
	rm -f $(DESTDIR)$(bindir)/par2create$(EXEEXT)
	rm -f $(DESTDIR)$(bindir)/par2verify$(EXEEXT)
	rm -f $(DESTDIR)$(bindir)/par2repair$(EXEEXT)

