//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2003 Peter Brian Clements
//  Copyright (c) 2019 Michael D. Nahas
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

#include <par2/cli.h>

#include "commandline.h"
// This is included here, so that cout and cerr are not used elsewhere.
#include <iostream>

#ifdef _WIN32
#include "wargs.h"
#endif

namespace par2
{

Result run(int argc, const char * const *argv, const Backends &backends)
{
  // Parse the command line
  CommandLine commandline;

  Result result = eInvalidCommandLineArguments;

  if (commandline.Parse(argc, argv))
  {
    // Which operation was selected
    switch (commandline.GetOperation())
    {
      case CommandLine::opCreate:
	// Create recovery data
	result = par2create(std::cout,
			    std::cerr,
			    commandline.GetNoiseLevel(),
			    commandline.GetMemoryLimit(),
			    commandline.GetBasePath(),
			    commandline.GetNumThreads(),
			    commandline.GetFileThreads(),
			    commandline.GetParFilename(),
			    commandline.GetExtraFiles(),

			    commandline.GetBlockSize(),

			    commandline.GetFirstRecoveryBlock(),
			    commandline.GetRecoveryFileScheme(),
			    commandline.GetRecoveryFileCount(),
			    commandline.GetRecoveryBlockCount(),
			    backends
			    );

        break;
      case CommandLine::opVerify:
      case CommandLine::opRepair:
        {
          // Verify or Repair damaged files
          switch (commandline.GetVersion())
          {
            case CommandLine::verPar1:
	      result = par1repair(std::cout,
				  std::cerr,
				  commandline.GetNoiseLevel(),
				  commandline.GetMemoryLimit(),
				  commandline.GetNumThreads(),
				  commandline.GetParFilename(),
				  commandline.GetExtraFiles(),
				  commandline.GetOperation() == CommandLine::opRepair,
				  commandline.GetPurgeFiles());

              break;
            case CommandLine::verPar2:
	      result = par2repair(std::cout,
				  std::cerr,
				  commandline.GetNoiseLevel(),
				  commandline.GetMemoryLimit(),
				  commandline.GetBasePath(),
				  commandline.GetNumThreads(),
				  commandline.GetFileThreads(),
				  commandline.GetParFilename(),
				  commandline.GetExtraFiles(),
				  commandline.GetOperation() == CommandLine::opRepair,
				  commandline.GetPurgeFiles(),
				  commandline.GetRenameOnly(),
				  commandline.GetSkipData(),
				  commandline.GetSkipLeaway(),
				  commandline.GetFullHash(),
				  backends);
              break;
	    default:
              break;
          }
        }
        break;
      case CommandLine::opNone:
        result = eSuccess;
        break;
      default:
        break;
    }
  }

  return result;
}

#ifdef _WIN32

Result run(int argc, wchar_t *wargv[], const Backends &backends)
{
  utf8::WideToUtf8ArgsAdapter wargsAdapter{ argc, wargv };

  return run(wargsAdapter.GetArgc(), wargsAdapter.GetUtf8Args(), backends);
}

#endif

} // namespace par2
