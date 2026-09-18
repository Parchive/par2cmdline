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

#ifndef __CLI_H__
#define __CLI_H__

#include <par2/backends.h>
#include <par2/libpar2.h>

namespace par2
{

// Carry out what the command line asks for: the arguments are parsed as the
// par2 tool parses them, and what is written to the output and error streams
// is what the tool writes. The result is what the tool returns as its exit
// code.
//
// Nothing about the process is changed: the tool's own main is what turns off
// iostream syncing with stdio and, on Windows, puts the console into UTF-8.
//
// backends holds the implementations the application supplies, each of which
// falls back to the one built in when it is left empty.
Result run(int argc, const char * const *argv, const Backends &backends = Backends());

#ifdef _WIN32

// The arguments a wide entry point is given, converted to UTF-8 before they
// are parsed. The names written out are UTF-8 too, so a console showing them
// wants SetConsoleOutputCP(CP_UTF8), as the tool calls.
Result run(int argc, wchar_t *wargv[], const Backends &backends = Backends());

#endif

} // namespace par2

#endif // __CLI_H__
