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

CommandLine::ExtraFile::ExtraFile(void)
: filename()
, filesize(0)
{
}

CommandLine::ExtraFile::ExtraFile(const CommandLine::ExtraFile &other)
: filename(other.filename)
, filesize(other.filesize)
{
}

CommandLine::ExtraFile& CommandLine::ExtraFile::operator=(const CommandLine::ExtraFile &other)
{
  filename = other.filename;
  filesize = other.filesize;

  return *this;
}

CommandLine::ExtraFile::ExtraFile(const string &name, u64 size)
: filename(name)
, filesize(size)
{
}


CommandLine::CommandLine(void)
: operation(opNone)
, version(verUnknown)
, noiselevel(nlUnknown)
, blockcount(0)
, blocksize(0)
, firstblock(0)
, recoveryfilescheme(scUnknown)
, recoveryfilecount(0)
, recoveryblockcount(0)
, recoveryblockcountset(false)
, redundancy(0)
, redundancysize(0)
, redundancyset(false)
, parfilename()
, extrafiles()
, totalsourcesize(0)
, largestsourcesize(0)
, memorylimit(0)
, purgefiles(false)
, recursive(false)
, skipdata(false)
, skipleaway(0)
{
}

void CommandLine::showversion(void)
{
  string version = PACKAGE " version " VERSION;
  cout << version << endl;
}

void CommandLine::banner(void)
{
  cout << "Copyright (C) 2003-2015 Peter Brian Clements." << endl
    << "Copyright (C) 2011-2012 Marcel Partap." << endl
    << "Copyright (C) 2012-2015 Ike Devolder." << endl
    << endl
    << "par2cmdline comes with ABSOLUTELY NO WARRANTY." << endl
    << endl
    << "This is free software, and you are welcome to redistribute it and/or modify" << endl
    << "it under the terms of the GNU General Public License as published by the" << endl
    << "Free Software Foundation; either version 2 of the License, or (at your" << endl
    << "option) any later version. See COPYING for details." << endl
    << endl;
}

void CommandLine::usage(void)
{
  cout <<
    "Usage:\n"
    "  par2 -h  : show this help\n"
    "  par2 -V  : show version\n"
    "  par2 -VV : show version and copyright\n"
    "\n"
    "  par2 c(reate) [options] <par2 file> [files] : Create PAR2 files\n"
    "  par2 v(erify) [options] <par2 file> [files] : Verify files using PAR2 file\n"
    "  par2 r(epair) [options] <par2 file> [files] : Repair files using PAR2 files\n"
    "\n"
    "You may also leave out the \"c\", \"v\", and \"r\" commands by using \"par2create\",\n"
    "\"par2verify\", or \"par2repair\" instead.\n"
    "\n"
    "Options:\n"
    "\n"
    "  -a<file> : Set the main par2 archive name\n"
    "  -b<n>    : Set the Block-Count\n"
    "  -s<n>    : Set the Block-Size (Don't use both -b and -s)\n"
    "  -r<n>    : Level of Redundancy (%%)\n"
    "  -r<c><n> : Redundancy target size, <c>=g(iga),m(ega),k(ilo) bytes\n"
    "  -c<n>    : Recovery block count (Don't use both -r and -c)\n"
    "  -f<n>    : First Recovery-Block-Number\n"
    "  -u       : Uniform recovery file sizes\n"
    "  -l       : Limit size of recovery files (Don't use both -u and -l)\n"
    "  -n<n>    : Number of recovery files (Don't use both -n and -l)\n"
    "  -m<n>    : Memory (in MB) to use\n"
    "  -v [-v]  : Be more verbose\n"
    "  -q [-q]  : Be more quiet (-q -q gives silence)\n"
    "  -p       : Purge backup files and par files on successful recovery or\n"
    "             when no recovery is needed\n"
    "  -R       : Recurse into subdirectories (only useful on create)\n"
    "  -N       : data skipping (find badly mispositioned data blocks)\n"
    "  -S<n>    : Skip leaway (distance +/- from expected block position)\n"
    "  -B<path> : Set the basepath to use as reference for the datafiles\n"
    "  --       : Treat all remaining CommandLine as filenames\n"
    "\n";
}

bool CommandLine::Parse(int argc, char *argv[])
{
  if (argc<1)
  {
    return false;
  }

  // Split the program name into path and filename
  string path, name;
  DiskFile::SplitFilename(argv[0], path, name);
  argc--;
  argv++;

  basepath = DiskFile::GetCanonicalPathname("./");

  if (argc>0)
  {
    if (argv[0][0] != 0 &&
        argv[0][0] == '-')
    {
      if (argv[0][1] != 0)
      {
        switch (argv[0][1])
        {
        case 'h':
          usage();
          return true;
        case 'V':
          showversion();
          if (argv[0][2] != 0 &&
              argv[0][2] == 'V')
          {
            cout << endl;
            banner();
          }
          return true;
        case '-':
          if (0 == stricmp(argv[0], "--help"))
          {
            usage();
            return true;
          }
        }
      }
    }
  }

  // Strip ".exe" from the end
  if (name.size() > 4 && 0 == stricmp(".exe", name.substr(name.length()-4).c_str()))
  {
    name = name.substr(0, name.length()-4);
  }

  // Check the resulting program name
  if (0 == stricmp("par2create", name.c_str()))
  {
    operation = opCreate;
  }
  else if (0 == stricmp("par2verify", name.c_str()))
  {
    operation = opVerify;
  }
  else if (0 == stricmp("par2repair", name.c_str()))
  {
    operation = opRepair;
  }

  // Have we determined what operation we want?
  if (operation == opNone)
  {
    if (argc<2)
    {
      cerr << "Not enough command line arguments." << endl;
      return false;
    }

    switch (tolower(argv[0][0]))
    {
    case 'c':
      if (argv[0][1] == 0 || 0 == stricmp(argv[0], "create"))
        operation = opCreate;
      break;
    case 'v':
      if (argv[0][1] == 0 || 0 == stricmp(argv[0], "verify"))
        operation = opVerify;
      break;
    case 'r':
      if (argv[0][1] == 0 || 0 == stricmp(argv[0], "repair"))
        operation = opRepair;
      break;
    }

    if (operation == opNone)
    {
      cerr << "Invalid operation specified: " << argv[0] << endl;
      return false;
    }
    argc--;
    argv++;
  }

  bool options = true;
  list<string> a_filenames;

  while (argc>0)
  {
    if (argv[0][0])
    {
      if (options && argv[0][0] != '-')
        options = false;

      if (options)
      {
        switch (argv[0][1])
        {
        case 'a':
          {
            if (operation == opCreate)
            {
              string str = argv[0];
              if (str == "-a")
              {
                SetParFilename(argv[1]);
                argc--;
                argv++;
              }
              else
              {
                SetParFilename(str.substr(2));
              }
            }
          }
          break;
        case 'b':  // Set the block count
          {
            if (operation != opCreate)
            {
              cerr << "Cannot specify block count unless creating." << endl;
              return false;
            }
            if (blockcount > 0)
            {
              cerr << "Cannot specify block count twice." << endl;
              return false;
            }
            else if (blocksize > 0)
            {
              cerr << "Cannot specify both block count and block size." << endl;
              return false;
            }

            char *p = &argv[0][2];
            while (blockcount <= 3276 && *p && isdigit(*p))
            {
              blockcount = blockcount * 10 + (*p - '0');
              p++;
            }
            if (0 == blockcount || blockcount > 32768 || *p)
            {
              cerr << "Invalid block count option: " << argv[0] << endl;
              return false;
            }
          }
          break;

        case 's':  // Set the block size
          {
            if (operation != opCreate)
            {
              cerr << "Cannot specify block size unless creating." << endl;
              return false;
            }
            if (blocksize > 0)
            {
              cerr << "Cannot specify block size twice." << endl;
              return false;
            }
            else if (blockcount > 0)
            {
              cerr << "Cannot specify both block count and block size." << endl;
              return false;
            }

            char *p = &argv[0][2];
            while (blocksize <= 429496729 && *p && isdigit(*p))
            {
              blocksize = blocksize * 10 + (*p - '0');
              p++;
            }
            if (*p || blocksize == 0)
            {
              cerr << "Invalid block size option: " << argv[0] << endl;
              return false;
            }
            if (blocksize & 3)
            {
              cerr << "Block size must be a multiple of 4." << endl;
              return false;
            }
          }
          break;

        case 'r':  // Set the amount of redundancy required
          {
            if (operation != opCreate)
            {
              cerr << "Cannot specify redundancy unless creating." << endl;
              return false;
            }
            if (redundancyset)
            {
              cerr << "Cannot specify redundancy twice." << endl;
              return false;
            }
            else if (recoveryblockcountset)
            {
              cerr << "Cannot specify both redundancy and recovery block count." << endl;
              return false;
            }

            if (argv[0][2] == 'k'
                || argv[0][2] == 'm'
                || argv[0][2] == 'g'
            )
            {
              char *p = &argv[0][3];
              while (*p && isdigit(*p))
              {
                redundancysize = redundancysize * 10 + (*p - '0');
                p++;
              }
              switch (argv[0][2])
              {
                case 'g':
                  redundancysize = redundancysize * 1024;
                case 'm':
                  redundancysize = redundancysize * 1024;
                case 'k':
                  redundancysize = redundancysize * 1024;
                  break;
              }
            }
            else
            {
              char *p = &argv[0][2];
              while (redundancy <= 10 && *p && isdigit(*p))
              {
                redundancy = redundancy * 10 + (*p - '0');
                p++;
              }
              if (redundancy > 100 || *p)
              {
                cerr << "Invalid redundancy option: " << argv[0] << endl;
                return false;
              }
              if (redundancy == 0 && recoveryfilecount > 0)
              {
                cerr << "Cannot set redundancy to 0 and file count > 0" << endl;
                return false;
              }
            }
            redundancyset = true;
          }
          break;

        case 'c': // Set the number of recovery blocks to create
          {
            if (operation != opCreate)
            {
              cerr << "Cannot specify recovery block count unless creating." << endl;
              return false;
            }
            if (recoveryblockcountset)
            {
              cerr << "Cannot specify recovery block count twice." << endl;
              return false;
            }
            else if (redundancyset)
            {
              cerr << "Cannot specify both recovery block count and redundancy." << endl;
              return false;
            }

            char *p = &argv[0][2];
            while (recoveryblockcount <= 32768 && *p && isdigit(*p))
            {
              recoveryblockcount = recoveryblockcount * 10 + (*p - '0');
              p++;
            }
            if (recoveryblockcount > 32768 || *p)
            {
              cerr << "Invalid recoveryblockcount option: " << argv[0] << endl;
              return false;
            }
            if (recoveryblockcount == 0 && recoveryfilecount > 0)
            {
              cerr << "Cannot set recoveryblockcount to 0 and file count > 0" << endl;
              return false;
            }
            recoveryblockcountset = true;
          }
          break;

        case 'f':  // Specify the First block recovery number
          {
            if (operation != opCreate)
            {
              cerr << "Cannot specify first block number unless creating." << endl;
              return false;
            }
            if (firstblock > 0)
            {
              cerr << "Cannot specify first block twice." << endl;
              return false;
            }

            char *p = &argv[0][2];
            while (firstblock <= 3276 && *p && isdigit(*p))
            {
              firstblock = firstblock * 10 + (*p - '0');
              p++;
            }
            if (firstblock > 32768 || *p)
            {
              cerr << "Invalid first block option: " << argv[0] << endl;
              return false;
            }
          }
          break;

        case 'u':  // Specify uniformly sized recovery files
          {
            if (operation != opCreate)
            {
              cerr << "Cannot specify uniform files unless creating." << endl;
              return false;
            }
            if (argv[0][2])
            {
              cerr << "Invalid option: " << argv[0] << endl;
              return false;
            }
            if (recoveryfilescheme != scUnknown)
            {
              cerr << "Cannot specify two recovery file size schemes." << endl;
              return false;
            }

            recoveryfilescheme = scUniform;
          }
          break;

        case 'l':  // Limit the size of the recovery files
          {
            if (operation != opCreate)
            {
              cerr << "Cannot specify limit files unless creating." << endl;
              return false;
            }
            if (argv[0][2])
            {
              cerr << "Invalid option: " << argv[0] << endl;
              return false;
            }
            if (recoveryfilescheme != scUnknown)
            {
              cerr << "Cannot specify two recovery file size schemes." << endl;
              return false;
            }
            if (recoveryfilecount > 0)
            {
              cerr << "Cannot specify limited size and number of files at the same time." << endl;
              return false;
            }

            recoveryfilescheme = scLimited;
          }
          break;

        case 'n':  // Specify the number of recovery files
          {
            if (operation != opCreate)
            {
              cerr << "Cannot specify recovery file count unless creating." << endl;
              return false;
            }
            if (recoveryfilecount > 0)
            {
              cerr << "Cannot specify recovery file count twice." << endl;
              return false;
            }
            if (redundancyset && redundancy == 0)
            {
              cerr << "Cannot set file count when redundancy is set to 0." << endl;
              return false;
            }
            if (recoveryblockcountset && recoveryblockcount == 0)
            {
              cerr << "Cannot set file count when recovery block count is set to 0." << endl;
              return false;
            }
            if (recoveryfilescheme == scLimited)
            {
              cerr << "Cannot specify limited size and number of files at the same time." << endl;
              return false;
            }

            char *p = &argv[0][2];
            while (*p && isdigit(*p))
            {
              recoveryfilecount = recoveryfilecount * 10 + (*p - '0');
              p++;
            }
            if (recoveryfilecount == 0 || *p)
            {
              cerr << "Invalid recovery file count option: " << argv[0] << endl;
              return false;
            }
          }
          break;

        case 'm':  // Specify how much memory to use for output buffers
          {
            if (memorylimit > 0)
            {
              cerr << "Cannot specify memory limit twice." << endl;
              return false;
            }

            char *p = &argv[0][2];
            while (*p && isdigit(*p))
            {
              memorylimit = memorylimit * 10 + (*p - '0');
              p++;
            }
            if (memorylimit == 0 || *p)
            {
              cerr << "Invalid memory limit option: " << argv[0] << endl;
              return false;
            }
          }
          break;

        case 'v':
          {
            switch (noiselevel)
            {
            case nlUnknown:
              {
                if (argv[0][2] == 'v')
                  noiselevel = nlDebug;
                else
                  noiselevel = nlNoisy;
              }
              break;
            case nlNoisy:
            case nlDebug:
              noiselevel = nlDebug;
              break;
            default:
              cerr << "Cannot use both -v and -q." << endl;
              return false;
              break;
            }
          }
          break;

        case 'q':
          {
            switch (noiselevel)
            {
            case nlUnknown:
              {
                if (argv[0][2] == 'q')
                  noiselevel = nlSilent;
                else
                  noiselevel = nlQuiet;
              }
              break;
            case nlQuiet:
            case nlSilent:
              noiselevel = nlSilent;
              break;
            default:
              cerr << "Cannot use both -v and -q." << endl;
              return false;
              break;
            }
          }
          break;

        case 'p':
          {
            if (operation != opRepair && operation != opVerify)
            {
              cerr << "Cannot specify purge unless repairing or verifying." << endl;
              return false;
            }
            purgefiles = true;
          }
          break;

        case 'h':
          {
            usage();
            return false;
            break;
          }

        case 'R':
          {
            if (operation == opCreate)
            {
              recursive = true;
            }
            else
            {
              cerr << "Recursive has no impact except on creating." << endl;
            }
          }
          break;

        case 'N':
          {
            if (operation == opCreate)
            {
              cerr << "Cannot specify Data Skipping unless reparing or verifying." << endl;
              return false;
            }
            skipdata = true;
          }
          break;

        case 'S':  // Set the skip leaway
          {
            if (operation == opCreate)
            {
              cerr << "Cannot specify skip leaway when creating." << endl;
              return false;
            }
            if (!skipdata)
            {
              cerr << "Cannot specify skip leaway and no skipping." << endl;
              return false;
            }

            char *p = &argv[0][2];
            while (skipleaway <= 429496729 && *p && isdigit(*p))
            {
              skipleaway = skipleaway * 10 + (*p - '0');
              p++;
            }
            if (*p || skipleaway == 0)
            {
              cerr << "Invalid skipleaway option: " << argv[0] << endl;
              return false;
            }
          }
          break;

        case 'B': // Set the basepath manually
          {
            string str = argv[0];
            if (str == "-B")
            {
              basepath = DiskFile::GetCanonicalPathname(argv[1]);
              argc--;
              argv++;
            }
            else
            {
              basepath = DiskFile::GetCanonicalPathname(str.substr(2));
            }
            string lastchar = basepath.substr(basepath.length() -1);
            if ("/" != lastchar && "\\" != lastchar)
            {
#ifdef WIN32
              basepath = basepath + "\\";
#else
              basepath = basepath + "/";
#endif
            }
          }
          break;

        case '-':
          {
            argc--;
            argv++;
            options = false;
            continue;
          }
          break;
        default:
          {
            cerr << "Invalid option specified: " << argv[0] << endl;
            return false;
          }
        }
      }
      else if (parfilename.length() == 0)
      {
        string filename = argv[0];
        string::size_type where;
        if ((where = filename.find_first_of('*')) != string::npos ||
            (where = filename.find_first_of('?')) != string::npos)
        {
          cerr << "par2 file must not have a wildcard in it." << endl;
          return false;
        }

        SetParFilename(filename);
      }
      else
      {
        list<string> *filenames;

        string path;
        string name;
        DiskFile::SplitFilename(argv[0], path, name);
        filenames = DiskFile::FindFiles(path, name, recursive);

        list<string>::iterator fn = filenames->begin();
        while (fn != filenames->end())
        {
          // Convert filename from command line into a full path + filename
          string filename = DiskFile::GetCanonicalPathname(*fn);

          // Originally, all specified files were supposed to exist, or the program
          // would stop with an error message. This was not practical, for example in
          // a directory with files appearing and disappearing (an active download directory).
          // So the new rule is: when a specified file doesn't exist, it is silently skipped.
          if (!DiskFile::FileExists(filename))
          {
            cout << "Ignoring non-existent source file: " << filename << endl;
          }
          // skip files outside basepath
          else if (filename.find(basepath) == string::npos)
          {
                cout << "Ignoring out of basepath source file: " << filename << endl;
          }
          else
          {
            u64 filesize = DiskFile::GetFileSize(filename);

            // Ignore all 0 byte files
            if (filesize == 0)
            {
              cout << "Skipping 0 byte file: " << filename << endl;
            }
            else if (a_filenames.end() != find(a_filenames.begin(), a_filenames.end(), filename))
            {
              cout << "Skipping duplicate filename: " << filename << endl;
            }
            else
            {
              a_filenames.push_back(filename);
              extrafiles.push_back(ExtraFile(filename, filesize));

              // track the total size of the source files and how
              // big the largest one is.
              totalsourcesize += filesize;
              if (largestsourcesize < filesize)
              largestsourcesize = filesize;
            }
          } //end file exists

          ++fn;
        }
        delete filenames;
      }
    }

    argc--;
    argv++;
  }

  if (parfilename.length() == 0)
  {
    cerr << "You must specify a Recovery file." << endl;
    return false;
  }

  // Default noise level
  if (noiselevel == nlUnknown)
  {
    noiselevel = nlNormal;
  }

  // Default skip leaway
  if (operation != opCreate
      && skipdata
      && skipleaway == 0)
  {
    // Expect to find blocks within +/- 64 bytes of the expected
    // position relative to the last block that was found.
    skipleaway = 64;
  }

  // If we a creating, check the other parameters
  if (operation == opCreate)
  {
    // If no recovery file size scheme is specified then use Variable
    if (recoveryfilescheme == scUnknown)
    {
      recoveryfilescheme = scVariable;
    }

    // If neither block count not block size is specified
    if (blockcount == 0 && blocksize == 0)
    {
      // Use a block count of 2000
      blockcount = 2000;
    }

    // If we are creating, the source files must be given.
    if (extrafiles.size() == 0)
    {
      // Does the par filename include the ".par2" on the end?
      if (parfilename.length() > 5 && 0 == stricmp(parfilename.substr(parfilename.length()-5, 5).c_str(), ".par2"))
      {
        // Yes it does.
        cerr << "You must specify a list of files when creating." << endl;
        return false;
      }
      else
      {
        // No it does not.

        // In that case check to see if the file exists, and if it does
        // assume that you wish to create par2 files for it.

        u64 filesize = 0;
        if (DiskFile::FileExists(parfilename) &&
            (filesize = DiskFile::GetFileSize(parfilename)) > 0)
        {
          extrafiles.push_back(ExtraFile(parfilename, filesize));

          // track the total size of the source files and how
          // big the largest one is.
          totalsourcesize += filesize;
          if (largestsourcesize < filesize)
            largestsourcesize = filesize;
        }
        else
        {
          // The file does not exist or it is empty.

          cerr << "You must specify a list of files when creating." << endl;
          return false;
        }
      }
    }

    // Strip the ".par2" from the end of the filename of the main PAR2 file.
    if (parfilename.length() > 5 && 0 == stricmp(parfilename.substr(parfilename.length()-5, 5).c_str(), ".par2"))
    {
      parfilename = parfilename.substr(0, parfilename.length()-5);
    }

    // Assume a redundancy of 5% if neither redundancy or recoveryblockcount were set.
    if (!redundancyset && !recoveryblockcountset)
    {
      redundancy = 5;
    }
  }

  // Assume a memory limit of 16MB if not specified.
  if (memorylimit == 0)
  {
#ifdef WIN32
    u64 TotalPhysicalMemory = 0;

    HMODULE hLib = ::LoadLibraryA("kernel32.dll");
    if (NULL != hLib)
    {
      BOOL (WINAPI *pfn)(LPMEMORYSTATUSEX) = (BOOL (WINAPI*)(LPMEMORYSTATUSEX))::GetProcAddress(hLib, "GlobalMemoryStatusEx");

      if (NULL != pfn)
      {
        MEMORYSTATUSEX mse;
        mse.dwLength = sizeof(mse);
        if (pfn(&mse))
        {
          TotalPhysicalMemory = mse.ullTotalPhys;
        }
      }

      ::FreeLibrary(hLib);
    }

    if (TotalPhysicalMemory == 0)
    {
      MEMORYSTATUS ms;
      ::ZeroMemory(&ms, sizeof(ms));
      ::GlobalMemoryStatus(&ms);

      TotalPhysicalMemory = ms.dwTotalPhys;
    }

    if (TotalPhysicalMemory == 0)
    {
      // Assume 128MB
      TotalPhysicalMemory = 128 * 1048576;
    }

    // Half of total physical memory
    memorylimit = (size_t)(TotalPhysicalMemory / 1048576 / 2);
#else
    memorylimit = 16;
#endif
  }
  memorylimit *= 1048576;

  return true;
}

void CommandLine::SetParFilename(string filename)
{
  parfilename = DiskFile::GetCanonicalPathname(filename);

  // If we are verifying or repairing, the PAR2 file must
  // already exist
  if (operation != opCreate)
  {
    // Find the last '.' in the filename
    string::size_type where = filename.find_last_of('.');
    if (where != string::npos)
    {
      // Get what follows the last '.'
      string tail = filename.substr(where+1);

      if (0 == stricmp(tail.c_str(), "par2"))
      {
        parfilename = filename;
        version = verPar2;
      }
      else if (0 == stricmp(tail.c_str(), "par") ||
               (tail.size() == 3 &&
               tolower(tail[0]) == 'p' &&
               isdigit(tail[1]) &&
               isdigit(tail[2])))
      {
        parfilename = filename;
        version = verPar1;
      }
    }

    // If we haven't figured out which version of PAR file we
    // are using from the file extension, then presumable the
    // files filename was actually the name of a data file.
    if (version == verUnknown)
    {
      // Check for the existence of a PAR2 of PAR file.
      if (DiskFile::FileExists(filename + ".par2"))
      {
        version = verPar2;
        parfilename = filename + ".par2";
      }
      else if (DiskFile::FileExists(filename + ".PAR2"))
      {
        version = verPar2;
        parfilename = filename + ".PAR2";
      }
      else if (DiskFile::FileExists(filename + ".par"))
      {
        version = verPar1;
        parfilename = filename + ".par";
      }
      else if (DiskFile::FileExists(filename + ".PAR"))
      {
        version = verPar1;
        parfilename = filename + ".PAR";
      }
    }
    else
    {
      // Does the specified PAR or PAR2 file exist
      if (!DiskFile::FileExists(filename))
      {
        version = verUnknown;
      }
    }
  }
  else
  {
    version = verPar2;
  }
}
