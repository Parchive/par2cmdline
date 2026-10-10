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

#ifndef PAR2_CLI_H
#define PAR2_CLI_H

#include <ostream>

#include <par2/backends.h>
#include <par2/libpar2.h>

namespace par2
{

// Carry out what the command line asks for: the arguments are parsed as the
// par2 tool parses them, argv[0] being the name it was run as, and what the
// tool writes to its output and error streams is written to sout and serr.
// The result is what the tool returns as its exit code.
//
// A failure is reported through the result rather than thrown, whatever the
// work or the implementations the application supplies throw: running out of
// memory is eMemoryError, and any other exception eLogicError, each said on
// serr.
//
// Nothing about the process is changed: the tool's own main is what turns off
// iostream syncing with stdio and, on Windows, puts the console into UTF-8.
//
// backends holds the implementations the application supplies, each of which
// falls back to the one built in when it is left empty.
Result run(int argc, const char * const *argv, std::ostream &sout, std::ostream &serr,
           const Backends &backends = Backends());

// Stop the work of every run in progress, from any thread, as Cancel stops a
// handle's: each returns eCancelled, having removed the files a create or a
// repair was part way through writing. False when no run is doing work that
// can be cancelled, such as one reading PAR1 files or one yet to start.
//
// It takes a lock, so a POSIX signal handler must not call it. A thread which
// waits for the signal, as the tool's main does, can.
bool cancel(void);

#ifdef _WIN32

// The arguments a wide entry point is given, converted to UTF-8 before they
// are parsed. The names written out are UTF-8 too, so a console showing them
// wants SetConsoleOutputCP(CP_UTF8), as the tool calls.
Result run(int argc, wchar_t *wargv[], std::ostream &sout, std::ostream &serr,
           const Backends &backends = Backends());

#endif

} // namespace par2

#endif // PAR2_CLI_H
