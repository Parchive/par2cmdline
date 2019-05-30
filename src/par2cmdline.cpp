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

#include "libpar2.h"
#include "commandline.h"

// This is included here, so that cout and cerr are not used elsewhere.
#include <iostream>

#ifdef _MSC_VER
#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif
#endif

int main(int argc, char *argv[])
{
#ifdef _MSC_VER
  // Memory leak checking
  _CrtSetDbgFlag(_CrtSetDbgFlag(_CRTDBG_REPORT_FLAG) | _CRTDBG_ALLOC_MEM_DF | /*_CRTDBG_CHECK_CRT_DF | */_CRTDBG_DELAY_FREE_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

  // check sizeof integers
  static_assert(sizeof(u8) == 1 || sizeof(i8) == 1
		|| sizeof(u16) == 2 || sizeof(i16) == 1
		|| sizeof(u32) == 4 || sizeof(i32) == 1
		|| sizeof(u64) == 8 || sizeof(i64) == 1,
		"Error: the assumed sizes of integers is wrong!");

  
  // Parse the command line
  CommandLine *commandline = new CommandLine;

  Result result = eInvalidCommandLineArguments;

  if (commandline->Parse(argc, argv))
  {
    // Which operation was selected
    switch (commandline->GetOperation())
    {
      case CommandLine::opCreate:
	// Create recovery data
	result = par2create(std::cout,
			    std::cerr,
			    commandline->GetNoiseLevel(),
			    commandline->GetMemoryLimit(),
			    commandline->GetBasePath(),
#ifdef _OPENMP
			    commandline->GetNumThreads(),
			    commandline->GetFileThreads(),
#endif
			    commandline->GetParFilename(),
			    commandline->GetExtraFiles(),

			    commandline->GetBlockSize(),
			    
			    commandline->GetFirstRecoveryBlock(),
			    commandline->GetRecoveryFileScheme(),
			    commandline->GetRecoveryFileCount(),
			    commandline->GetRecoveryBlockCount()
			    );

        break;
      case CommandLine::opVerify:
      case CommandLine::opRepair:
        {
          // Verify or Repair damaged files
          switch (commandline->GetVersion())
          {
            case CommandLine::verPar1:
	      result = par1repair(std::cout,
				  std::cerr,
				  commandline->GetNoiseLevel(),
				  commandline->GetMemoryLimit(),
#ifdef _OPENMP
				  commandline->GetNumThreads(),
#endif
				  commandline->GetParFilename(),
				  commandline->GetExtraFiles(),
				  commandline->GetOperation() == CommandLine::opRepair,
				  commandline->GetPurgeFiles());
	      
              break;
            case CommandLine::verPar2:
	      result = par2repair(std::cout,
				  std::cerr,
				  commandline->GetNoiseLevel(),
				  commandline->GetMemoryLimit(),
				  commandline->GetBasePath(),
#ifdef _OPENMP
				  commandline->GetNumThreads(),
				  commandline->GetFileThreads(),
#endif
				  commandline->GetParFilename(),
				  commandline->GetExtraFiles(),
				  commandline->GetOperation() == CommandLine::opRepair,
				  commandline->GetPurgeFiles(),
				  commandline->GetSkipData(),
				  commandline->GetSkipLeaway());
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

  delete commandline;

  return result;
}
