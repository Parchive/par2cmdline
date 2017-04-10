//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See http://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2003 Peter Brian Clements
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

#include "par2cmdline.h"

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

  // Parse the command line
  CommandLine *commandline = new CommandLine;

  Result result = eInvalidCommandLineArguments;

  if (commandline->Parse(argc, argv))
  {
    // Which operation was selected
    switch (commandline->GetOperation())
    {
      case CommandLine::opCreate:
        {
          // Create recovery data

          Par2Creator *creator = new Par2Creator;
          result = creator->Process(*commandline);
          delete creator;
        }
        break;
      case CommandLine::opVerify:
        {
          // Verify damaged files
          switch (commandline->GetVersion())
          {
            case CommandLine::verPar1:
              {
                Par1Repairer *repairer = new Par1Repairer;
                result = repairer->Process(*commandline, false);
                delete repairer;
              }
              break;
            case CommandLine::verPar2:
              {
                Par2Repairer *repairer = new Par2Repairer;
                result = repairer->Process(*commandline, false);
                delete repairer;
              }
              break;
            case CommandLine::opNone:
              break;
          }
        }
        break;
      case CommandLine::opRepair:
        {
          // Repair damaged files
          switch (commandline->GetVersion())
          {
            case CommandLine::verPar1:
              {
                Par1Repairer *repairer = new Par1Repairer;
                result = repairer->Process(*commandline, true);
                delete repairer;
              }
              break;
            case CommandLine::verPar2:
              {
                Par2Repairer *repairer = new Par2Repairer;
                result = repairer->Process(*commandline, true);
                delete repairer;
              }
              break;
            case CommandLine::opNone:
              break;
          }
        }
        break;
      case CommandLine::opNone:
        result = eSuccess;
        break;
    }
  }

  delete commandline;

  return result;
}
