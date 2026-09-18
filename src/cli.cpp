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

#ifdef _WIN32
#include "wargs.h"
#endif

#include <new>

namespace par2
{

// Say on serr what the exception being handled stopped, and return it as the
// failure of the run
static Result Thrown(std::ostream &serr)
{
  try
  {
    throw;
  }
  catch (const std::bad_alloc &)
  {
    serr << "Memory ran out." << std::endl;
    return eMemoryError;
  }
  catch (...)
  {
    serr << "The work stopped on an exception." << std::endl;
    return eLogicError;
  }
}

Result run(int argc, const char * const *argv, std::ostream &sout, std::ostream &serr,
           const Backends &backends)
try
{
  // Parse the command line
  CommandLine commandline(sout, serr);

  Result result = eInvalidCommandLineArguments;

  if (commandline.Parse(argc, argv))
  {
    // Which operation was selected
    switch (commandline.GetOperation())
    {
      case CommandLine::opCreate:
        // Create recovery data
        result = par2create(sout,
                            serr,
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
              result = par1repair(sout,
                                  serr,
                                  commandline.GetNoiseLevel(),
                                  commandline.GetMemoryLimit(),
                                  commandline.GetNumThreads(),
                                  commandline.GetParFilename(),
                                  commandline.GetExtraFiles(),
                                  commandline.GetOperation() == CommandLine::opRepair,
                                  commandline.GetPurgeFiles());

              break;
            case CommandLine::verPar2:
              result = par2repair(sout,
                                  serr,
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
catch (...)
{
  return Thrown(serr);
}

#ifdef _WIN32

Result run(int argc, wchar_t *wargv[], std::ostream &sout, std::ostream &serr,
           const Backends &backends)
try
{
  utf8::WideToUtf8ArgsAdapter wargsAdapter{ argc, wargv, serr };

  return run(wargsAdapter.GetArgc(), wargsAdapter.GetUtf8Args(), sout, serr, backends);
}
catch (...)
{
  return Thrown(serr);
}

#endif

} // namespace par2
