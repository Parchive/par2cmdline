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

// This is included here, so that cout and cerr are not used elsewhere.
#include<iostream>
#include<algorithm>
#include "commandline.h"
using namespace std;

#ifdef _MSC_VER
#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif
#endif

// OpenMP
#ifdef _OPENMP
# include <omp.h>
#endif

CommandLine::CommandLine(void)
: filesize_cache()
, version(verUnknown)
, noiselevel(nlUnknown)
, memorylimit(0)
, basepath()
#ifdef _OPENMP
, nthreads(0) // 0 means use default number
, filethreads( _FILE_THREADS ) // default from header file
#endif
, parfilename()
, rawfilenames()
, extrafiles()
, operation(opNone)  
, purgefiles(false)
, skipdata(false)
, skipleaway(0)
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
, recursive(false)
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
    << "Copyright (C) 2012-2017 Ike Devolder." << endl
    << "Copyright (C) 2014-2017 Jussi Kansanen." << endl
    << "Copyright (C) 2019 Michael Nahas." << endl
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
    "  par2 c(reate) [options] <PAR2 file> [files] : Create PAR2 files\n"
    "  par2 v(erify) [options] <PAR2 file> [files] : Verify files using PAR2 file\n"
    "  par2 r(epair) [options] <PAR2 file> [files] : Repair files using PAR2 files\n"
    "\n"
    "You may also leave out the \"c\", \"v\", and \"r\" commands by using \"par2create\",\n"
    "\"par2verify\", or \"par2repair\" instead.\n"
    "\n"
    "Options: (all uses)\n"
    "  -B<path> : Set the basepath to use as reference for the datafiles\n"
    "  -v [-v]  : Be more verbose\n"
    "  -q [-q]  : Be more quiet (-q -q gives silence)\n"
    "  -m<n>    : Memory (in MB) to use\n";
#ifdef _OPENMP
  cout <<
    "  -t<n>    : Number of threads used for main processing (" << omp_get_max_threads() << " detected)\n"
    "  -T<n>    : Number of files hashed in parallel\n"
    "             (" << _FILE_THREADS << " are the default)\n";
#endif
  cout <<
    "  --       : Treat all following arguments as filenames\n"
    "Options: (verify or repair)\n"
    "  -p       : Purge backup files and par files on successful recovery or\n"
    "             when no recovery is needed\n"
    "  -N       : Data skipping (find badly mispositioned data blocks)\n"
    "  -S<n>    : Skip leaway (distance +/- from expected block position)\n"
    "Options: (create)\n"
    "  -a<file> : Set the main PAR2 archive name\n"
    "  -b<n>    : Set the Block-Count\n"
    "  -s<n>    : Set the Block-Size (don't use both -b and -s)\n"
    "  -r<n>    : Level of redundancy (%%)\n"
    "  -r<c><n> : Redundancy target size, <c>=g(iga),m(ega),k(ilo) bytes\n"
    "  -c<n>    : Recovery Block-Count (don't use both -r and -c)\n"
    "  -f<n>    : First Recovery-Block-Number\n"
    "  -u       : Uniform recovery file sizes\n"
    "  -l       : Limit size of recovery files (don't use both -u and -l)\n"
    "  -n<n>    : Number of recovery files (don't use both -n and -l)\n"
    "  -R       : Recurse into subdirectories\n"
    "\n";
  cout <<
    "Example:\n"
    "   par2 repair *.par2\n"
    "\n";
}

bool CommandLine::Parse(int argc, const char * const *argv)
{
  if (!ReadArgs(argc, argv))
    return false;

  if (operation != opNone) {  // user didn't do "par --help", etc.
    if (!CheckValuesAndSetDefaults())
      return false;
  }

  if (operation == opCreate) {
    if (!ComputeBlockSize())
      return false;

    u64 sourceblockcount = 0;
    u64 largestfilesize = 0;
    for (vector<string>::const_iterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
    {
      u64 filesize = filesize_cache.get(*i);
      sourceblockcount += (filesize + blocksize-1) / blocksize;
      if (filesize > largestfilesize)
      {
	largestfilesize = filesize;
      }
    }

    if (!ComputeRecoveryBlockCount(&recoveryblockcount,
				   sourceblockcount,
				   blocksize,
				   firstblock,
				   recoveryfilescheme,
				   recoveryfilecount,
				   recoveryblockcountset,
				   redundancy,
				   redundancysize,
				   largestfilesize))
    {
      return false;
    }
  }
  
  return true;
}

bool CommandLine::ReadArgs(int argc, const char * const *argv)
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

  if (argc>0)
  {
    if (argv[0][0] == '-')
    {
      if (argv[0] == string("-h") || argv[0] == string("--help"))
      {
	usage();
	return true;
      }
      else if (argv[0] == string("-V") || argv[0] == string("--version"))
      {
	showversion();
	return true;
      }
      else if (argv[0] == string("-VV"))
      {
	showversion();
	cout << endl;
	banner();
	return true;
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
  basepath = "";

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
              bool setparfile = false;
              if (str == "-a")
              {
                setparfile = SetParFilename(argv[1]);
                argc--;
                argv++;
              }
              else
              {
                setparfile = SetParFilename(str.substr(2));
              }

              if (! setparfile)
              {
                cerr << "failed to set the main par file" << endl;
                return false;
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

            const char *p = &argv[0][2];
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

            const char *p = &argv[0][2];
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

#ifdef _OPENMP
        case 't':  // Set amount of threads
          {
            nthreads = 0;
            const char *p = &argv[0][2];

            while (*p && isdigit(*p))
            {
              nthreads = nthreads * 10 + (*p - '0');
              p++;
            }

            if (!nthreads)
            {
              cerr << "Invalid thread option: " << argv[0] << endl;
              return false;
            }
          }
          break;

        case 'T':  // Set amount of file threads
          {
            filethreads = 0;
            const char *p = &argv[0][2];

            while (*p && isdigit(*p))
            {
              filethreads = filethreads * 10 + (*p - '0');
              p++;
            }

            if (!filethreads)
            {
              cerr << "Invalid file-thread option: " << argv[0] << endl;
              return false;
            }
          }
          break;
#endif

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
              const char *p = &argv[0][3];
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
              const char *p = &argv[0][2];
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

            const char *p = &argv[0][2];
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

            const char *p = &argv[0][2];
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
            // (Removed "Cannot set file count when redundancy is set to 0.")
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

            const char *p = &argv[0][2];
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

            const char *p = &argv[0][2];
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
          }
	  // "break;" not needed.

        case 'R':
          {
            if (operation == opCreate)
            {
              recursive = true;
            }
            else
            {
              cerr << "Cannot specific Recursive unless creating." << endl;
              return false;
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

            const char *p = &argv[0][2];
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
          }
          break;

        case '-':
          {
	    if (argv[0] != string("--")) {
              cerr << "Unknown option: " << argv[0] << endl;
	      cerr << "  (Options must appear after create, repair or verify.)" << endl;
	      cerr << "  (Run \"" << path << name << " --help\" for supported options.)" << endl;
              return false;
            }
	      
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
        bool setparfile = SetParFilename(filename);
        if (! setparfile)
        {
          cerr << "failed to set the main par file" << endl;
          return false;
        }
      }
      else
      {

        string path;
        string name;
        DiskFile::SplitFilename(argv[0], path, name);
	std::unique_ptr< list<string> > filenames(
						DiskFile::FindFiles(path, name, recursive)
						);

        list<string>::iterator fn = filenames->begin();
        while (fn != filenames->end())
        {
          // Convert filename from command line into a full path + filename
          string filename = DiskFile::GetCanonicalPathname(*fn);
          rawfilenames.push_back(filename);
          ++fn;
        }

        // delete filenames;   Taken care of by unique_ptr<>
      }
    }

    argc--;
    argv++;
  }

  return true;
}


// This webpage has code to get physical memory size on many OSes
// http://nadeausoftware.com/articles/2012/09/c_c_tip_how_get_physical_memory_size_system

#ifdef _WIN32
u64 CommandLine::GetTotalPhysicalMemory()
{
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

  return TotalPhysicalMemory;
}
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
// POSIX compliant OSes, including OSX/MacOS and Cygwin.  Also works for Linux.
u64 CommandLine::GetTotalPhysicalMemory()
{
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGESIZE);  
  return pages*page_size;
}
#else
// default version == unable to request memory size
u64 CommandLine::GetTotalPhysicalMemory()
{
  return 0;
}
#endif


bool CommandLine::CheckValuesAndSetDefaults() {
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

  // Default memorylimit of 128MB
  if (memorylimit == 0)
  {
    u64 TotalPhysicalMemory = GetTotalPhysicalMemory();

    if (TotalPhysicalMemory == 0)
    {
      // Default/error case:
      // NOTE: In 2019, Ubuntu's minimum requirements are 256MiB.
      TotalPhysicalMemory = 256 * 1048576;
    }

    // Half of total physical memory
    memorylimit = (size_t)(TotalPhysicalMemory / 1048576 / 2);
  }
  // convert to megabytes
  memorylimit *= 1048576;

  if (noiselevel >= nlDebug)
  {
    cout << "[DEBUG] memorylimit: " << memorylimit << " bytes" << endl;
  }


  // Default basepath  (uses parfilename)
  if ("" == basepath)
  {
    if (noiselevel >= nlDebug)
    {
      cout << "[DEBUG] parfilename: " << parfilename << endl;
    }

    string dummy;
    string path;
    DiskFile::SplitFilename(parfilename, path, dummy);
    basepath = DiskFile::GetCanonicalPathname(path);

    // fallback
    if ("" == basepath)
    {
      basepath = DiskFile::GetCanonicalPathname("./");
    }
  }

  string lastchar = basepath.substr(basepath.length() -1);
  if ("/" != lastchar && "\\" != lastchar)
  {
#ifdef _WIN32
    basepath = basepath + "\\";
#else
    basepath = basepath + "/";
#endif
  }

  if (noiselevel >= nlDebug)
  {
    cout << "[DEBUG] basepath: " << basepath << endl;
  }


  // parfilename is checked earlier, because it is used by basepath.


  // check extrafiles
  list<string>::iterator rawfilenames_fn;
  for (rawfilenames_fn = rawfilenames.begin(); rawfilenames_fn != rawfilenames.end(); ++rawfilenames_fn)
  {
    string filename = *rawfilenames_fn;

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
      u64 filesize = filesize_cache.get(filename);

      // Ignore all 0 byte files
      if (filesize == 0)
      {
        cout << "Skipping 0 byte file: " << filename << endl;
      }
      else if (extrafiles.end() != find(extrafiles.begin(), extrafiles.end(), filename))
      {
        cout << "Skipping duplicate filename: " << filename << endl;
      }
      else
      {
        extrafiles.push_back(filename);
      }
    } //end file exists
  }

  
  // operation should alway be set, but let's be thorough.
  if (operation == opNone) {
    cerr << "ERROR: No operation was specified (create, repair, or verify)" << endl;
    return false;
  }
  

  if (operation != opCreate) {
    // skipdata is bool and either value is valid.

    // Default skip leaway
    if (skipdata && skipleaway == 0)
    {
      // Expect to find blocks within +/- 64 bytes of the expected
      // position relative to the last block that was found.
      skipleaway = 64;
    }
  }

  // If we a creating, check the other parameters
  if (operation == opCreate)
  {
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
          extrafiles.push_back(parfilename);
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

    // If neither block count not block size is specified
    if (blockcount == 0 && blocksize == 0)
    {
      // Use a block count of 2000
      blockcount = 2000;
    }

    // If no recovery file size scheme is specified then use Variable
    if (recoveryfilescheme == scUnknown)
    {
      recoveryfilescheme = scVariable;
    }

    // Assume a redundancy of 5% if neither redundancy or recoveryblockcount were set.
    if (!redundancyset && !recoveryblockcountset)
    {
      redundancy = 5;
      redundancyset = true;
    }
  }


  return true;
}


bool CommandLine::ComputeBlockSize() {

  if (blocksize == 0) {
    // compute value from blockcount

    if (blockcount < extrafiles.size())
    {
      // The block count cannot be less than the number of files.

      cerr << "Block count (" << blockcount <<
              ") cannot be smaller than the number of files(" << extrafiles.size() << "). " << endl;
      return false;
    }
    else if (blockcount == extrafiles.size())
    {
      // If the block count is the same as the number of files, then the block
      // size is the size of the largest file (rounded up to a multiple of 4).

      u64 largestfilesize = 0;
      for (vector<string>::const_iterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
      {
	u64 filesize = filesize_cache.get(*i);
	if (filesize > largestfilesize)
	{
	  largestfilesize = filesize;
	}
      }
      blocksize = (largestfilesize + 3) & ~3;
    }
    else
    {
      u64 totalsize = 0;
      for (vector<string>::const_iterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
      {
        totalsize += (filesize_cache.get(*i) + 3) / 4;
      }

      if (blockcount > totalsize)
      {
        blocksize = 4;
      }
      else
      {
        // Absolute lower bound and upper bound on the source block size that will
        // result in the requested source block count.
        u64 lowerBound = totalsize / blockcount;
        u64 upperBound = (totalsize + blockcount - extrafiles.size() - 1) / (blockcount - extrafiles.size());

        u64 count = 0;
        u64 size;

        do
        {
          size = (lowerBound + upperBound)/2;

          count = 0;
          for (vector<string>::const_iterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
          {
            count += ((filesize_cache.get(*i)+3)/4 + size-1) / size;
          }
          if (count > blockcount)
          {
            lowerBound = size+1;
            if (lowerBound >= upperBound)
            {
              size = lowerBound;
              count = 0;
              for (vector<string>::const_iterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
              {
                count += ((filesize_cache.get(*i)+3)/4 + size-1) / size;
              }
            }
          }
          else
          {
            upperBound = size;
          }
        }
        while (lowerBound < upperBound);

        if (count > 32768)
        {
          cerr << "Error calculating block size. cannot be higher than 32768." << endl;
          return false;
        }
        else if (count == 0)
        {
          cerr << "Error calculating block size. cannot be 0." << endl;
          return false;
        }

        blocksize = size*4;
      }
    }
  }
  
  return true;
}


// Determine how many recovery blocks to create based on the source block
// count and the requested level of redundancy.
bool CommandLine::ComputeRecoveryBlockCount(u32 *recoveryblockcount,
					    u32 sourceblockcount,
					    u64 blocksize,
					    u32 firstblock,
					    Scheme recoveryfilescheme,
					    u32 recoveryfilecount,
					    bool recoveryblockcountset,
					    u32 redundancy,
					    u64 redundancysize,
					    u64 largestfilesize)
{
  if (recoveryblockcountset) {
    // no need to assign value.
    // pass through, so that value can be checked below.
  }
  else if (redundancy > 0)
  {
    // count is the number of input blocks

    // Determine recoveryblockcount
    *recoveryblockcount = (sourceblockcount * redundancy + 50) / 100;
  }
  else if (redundancysize > 0)
  {
    const u64 overhead_per_recovery_file = sourceblockcount * (u64) 21;
    const u64 recovery_packet_size = blocksize + (u64) 70;
    if (recoveryfilecount == 0)
    {
      u32 estimatedFileCount = 15;
      u64 overhead = estimatedFileCount * overhead_per_recovery_file;
      u64 estimatedrecoveryblockcount;
      if (overhead > redundancysize)
      {
        estimatedrecoveryblockcount = 1;  // at least 1
      }
      else
      {
	estimatedrecoveryblockcount = (u32)((redundancysize - overhead) / recovery_packet_size);
      }

      // recoveryfilecount assigned below.
      bool success = ComputeRecoveryFileCount(cout,
					      cerr,
					      &recoveryfilecount,
					      recoveryfilescheme,
					      estimatedrecoveryblockcount,
					      largestfilesize,
					      blocksize);
      if (!success) {
	return false;
      }
    }

    const u64 overhead = recoveryfilecount * overhead_per_recovery_file;
    if (overhead > redundancysize)
    {
      *recoveryblockcount = 1;  // at least 1
    }
    else
    {
      *recoveryblockcount = (u32)((redundancysize - overhead) / recovery_packet_size);
    }
  }
  else
  {
    cerr << "Redundancy and Redundancysize not set." << endl;
    return false;
  }

  // Force valid values if necessary
  if (*recoveryblockcount == 0 && redundancy > 0)
    *recoveryblockcount = 1;

  if (*recoveryblockcount > 65536)
  {
    cerr << "Too many recovery blocks requested." << endl;
    return false;
  }

  // Check that the last recovery block number would not be too large
  if (firstblock + *recoveryblockcount >= 65536)
  {
    cerr << "First recovery block number is too high." << endl;
    return false;
  }

  cout << endl;
  return true;
}





bool CommandLine::SetParFilename(string filename)
{
  bool result = false;
  string::size_type where;

  if ((where = filename.find_first_of('*')) != string::npos ||
      (where = filename.find_first_of('?')) != string::npos)
  {
    cerr << "par2 file must not have a wildcard in it." << endl;
    return result;
  }

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

      if (DiskFile::FileExists(filename)) {
        result = true;
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
        result = true;
      }
      else if (DiskFile::FileExists(filename + ".PAR2"))
      {
        version = verPar2;
        parfilename = filename + ".PAR2";
        result = true;
      }
      else if (DiskFile::FileExists(filename + ".par"))
      {
        version = verPar1;
        parfilename = filename + ".par";
        result = true;
      }
      else if (DiskFile::FileExists(filename + ".PAR"))
      {
        version = verPar1;
        parfilename = filename + ".PAR";
        result = true;
      }
    }
  }
  else
  {
    parfilename = DiskFile::GetCanonicalPathname(filename);
    version = verPar2;
    result = true;
  }

  return result;
}
