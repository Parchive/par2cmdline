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

bin_PROGRAMS = par2

par2_SOURCES = par2cmdline.cpp par2cmdline.h \
	commandline.cpp commandline.h \
	crc.cpp crc.h \
	creatorpacket.cpp creatorpacket.h \
	criticalpacket.cpp criticalpacket.h \
	datablock.cpp datablock.h \
	descriptionpacket.cpp descriptionpacket.h \
	diskfile.cpp diskfile.h \
	filechecksummer.cpp filechecksummer.h \
	galois.cpp galois.h \
	letype.h \
	mainpacket.cpp mainpacket.h \
	md5.cpp md5.h \
	par1fileformat.cpp par1fileformat.h \
	par1repairer.cpp par1repairer.h \
	par1repairersourcefile.cpp par1repairersourcefile.h \
	par2creator.cpp par2creator.h \
	par2creatorsourcefile.cpp par2creatorsourcefile.h \
	par2fileformat.cpp par2fileformat.h \
	par2repairer.cpp par2repairer.h \
	par2repairersourcefile.cpp par2repairersourcefile.h \
	recoverypacket.cpp recoverypacket.h \
	reedsolomon.cpp reedsolomon.h \
	verificationhashtable.cpp verificationhashtable.h \
	verificationpacket.cpp verificationpacket.h

LDADD = -lstdc++
AM_CXXFLAGS = -Wall

EXTRA_DIST = PORTING ROADMAP par2cmdline.sln par2cmdline.vcproj \
	testdata.tar.gz pretest test1 test2 test3 test4 test5 test6 \
	posttest

TESTS = pretest test1 test2 test3 test4 test5 test6 posttest

install-exec-hook :
	ln -f $(DESTDIR)$(bindir)/par2$(EXEEXT) $(DESTDIR)$(bindir)/par2create$(EXEEXT)
	ln -f $(DESTDIR)$(bindir)/par2$(EXEEXT) $(DESTDIR)$(bindir)/par2verify$(EXEEXT)
	ln -f $(DESTDIR)$(bindir)/par2$(EXEEXT) $(DESTDIR)$(bindir)/par2repair$(EXEEXT)

uninstall-hook :
	rm -f $(DESTDIR)$(bindir)/par2create$(EXEEXT)
	rm -f $(DESTDIR)$(bindir)/par2verify$(EXEEXT)
	rm -f $(DESTDIR)$(bindir)/par2repair$(EXEEXT)

