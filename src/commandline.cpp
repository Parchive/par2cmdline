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

#include "libpar2internal.h"

// iostream is included here, so that cout and cerr are not used elsewhere.
#include<iostream>
#include<algorithm>
#include "commandline.h"
#include "foreach_parallel.h"
#include <fstream>  //ADDED for @FILELIST FUNCTIONALY
#include <cxxopts.hpp>
#include <sstream>

#ifdef _MSC_VER
#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif
#else
#include <unistd.h>
#endif


namespace par2
{

// The number in text, false when it is not one or is above most
static bool ParseNumber(const std::string &text, const u64 most, u64 &number)
{
  if (text.empty())
    return false;

  number = 0;
  for (const char ch : text)
  {
    if (!isdigit((unsigned char)ch))
      return false;

    const u64 digit = (u64)(ch - '0');
    if (number > (most - digit) / 10)
      return false;

    number = number * 10 + digit;
  }

  return true;
}

// The options, which may come before or after the files, and the help each
// is listed with
static void AddOptions(cxxopts::Options &parser)
{
  parser.add_options("General")
    ("a,archive", "Set the main PAR2 archive name", cxxopts::value<std::string>(), "<file>")
    ("B,basepath", "Set the basepath to use as reference for the datafiles", cxxopts::value<std::string>(), "<path>")
    ("v,verbose", "Be more verbose (-v -v gives debug output)")
    ("q,quiet", "Be more quiet (-q -q gives silence)")
    ("m,memory", "Memory (in MB) to use (default is half of total physical memory)", cxxopts::value<std::string>(), "<n>")
    ("t,threads", "Number of threads used for main processing (" + std::to_string(default_threads()) + " detected)", cxxopts::value<std::string>(), "<n>")
    ("T,file-threads", "Number of files hashed in parallel (" + std::to_string(_FILE_THREADS) + " are the default)", cxxopts::value<std::string>(), "<n>")
    ("h,help", "Show this help")
    ("V,version", "Show version (-V -V adds the copyright)");

  parser.add_options("Verify and repair")
    ("p,purge", "Purge backup files and par files on successful recovery or when no recovery is needed")
    ("O,rename-only", "Rename-only mode (skip files that are not perfect matches, useful for quickly fixing renamed files)")
    ("N,data-skipping", "Data skipping (find badly mispositioned data blocks)")
    ("S,skip-leaway", "Skip leaway (distance +/- from expected block position, default 64)", cxxopts::value<std::string>(), "<n>")
    ("full-hash", "Also check the hash of the whole of each file");

  parser.add_options("Create")
    ("b,block-count", "Set the Block-Count (default 2000)", cxxopts::value<std::string>(), "<n>")
    ("s,block-size", "Set the Block-Size (don't use both -b and -s)", cxxopts::value<std::string>(), "<n>")
    ("r,redundancy", "Level of redundancy (%, default 5%)", cxxopts::value<std::string>(), "<n>")
    ("redundancy-size", "Total size of the recovery data, in bytes or with k, m or g after the number (don't use both -r and --redundancy-size)", cxxopts::value<std::string>(), "<n>")
    ("c,recovery-blocks", "Recovery Block-Count (don't use both -r and -c)", cxxopts::value<std::string>(), "<n>")
    ("f,first-recovery-block", "First Recovery-Block-Number (default 0)", cxxopts::value<std::string>(), "<n>")
    ("u,uniform", "Uniform recovery file sizes (default is variable)")
    ("l,limit", "Recovery files grow in size as by default, but none holds more than the largest source file (don't use both -u and -l)")
    ("n,recovery-files", "Number of recovery files (max 31) (don't use both -n and -l)", cxxopts::value<std::string>(), "<n>")
    ("R,recurse", "Recurse into subdirectories for the files a wildcard matches")
    ("L,follow-symlinks", "Follow symlinks to files when scanning for source files (use at your own risk: symlink targets may lie outside the basepath. Symlinks to directories are not followed. POSIX systems only)");

  parser.add_options()
    ("files", "", cxxopts::value<std::vector<std::string> >());
  parser.parse_positional("files");
}

CommandLine::CommandLine(void)
: CommandLine(std::cout, std::cerr)
{
}

CommandLine::CommandLine(std::ostream &_sout, std::ostream &_serr)
: sout(_sout)
, serr(_serr)
, filesize_cache()
, version(verUnknown)
, noiselevel(nlUnknown)
, memorylimit(0)
, basepath()
, nthreads(0) // 0 means use default number
, filethreads( _FILE_THREADS ) // default from header file
, parfilename()
, rawfilenames()
, extrafiles()
, operation(opNone)
, purgefiles(false)
, renameonly(false)
, skipdata(false)
, skipleaway(0)
, fullhash(false)
, blockcount(0)
, blocksize(0)
, firstblock(0)
, recoveryfilescheme(scUnknown)
, recoveryfilecount(0)
, recoveryblockcount(0)
, sourceblockcount(0)
, largestfilesize(0)
, recoveryblockcountset(false)
, redundancy(0)
, redundancysize(0)
, redundancyset(false)
, recursive(false)
, followlinks(false)
{
}

void CommandLine::showversion(void)
{
  std::string version = PACKAGE " version " VERSION;
  sout << version << std::endl;
}

void CommandLine::banner(void)
{
  sout << "Copyright (C) 2003-2015 Peter Brian Clements.\n"
    "Copyright (C) 2011-2012 Marcel Partap.\n"
    "Copyright (C) 2012-2026 Ike Devolder.\n"
    "Copyright (C) 2014-2017 Jussi Kansanen.\n"
    "Copyright (C) 2019 Michael Nahas.\n"
    "\n"
    "par2cmdline comes with ABSOLUTELY NO WARRANTY.\n"
    "\n"
    "This is free software, and you are welcome to redistribute it and/or modify\n"
    "it under the terms of the GNU General Public License as published by the\n"
    "Free Software Foundation; either version 2 of the License, or (at your\n"
    "option) any later version. See COPYING for details.\n"
    << std::endl;
}

void CommandLine::usage(void)
{
  sout <<
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
    "\n";

  cxxopts::Options parser("par2");
  AddOptions(parser);

  // Every group's descriptions start in one column, and wrap at the width
  const std::string::size_type column = 30;
  const std::string::size_type width = 79;

  for (const char *group : {"General", "Verify and repair", "Create"})
  {
    if (group[0] != 'G')
      sout << "\n";
    sout << group << " options:\n";

    for (const auto &option : parser.group_help(group).options)
    {
      std::string line = option.s.empty() ? "      " : "  -" + option.s + ", ";
      line += "--" + option.l.front();
      if (!option.is_boolean)
        line += " " + option.arg_help;

      if (line.size() + 2 > column)
      {
        sout << line << "\n";
        line.clear();
      }
      line.resize(column, ' ');

      std::istringstream words(option.desc);
      std::string word;
      bool first = true;
      while (words >> word)
      {
        if (!first && line.size() + 1 + word.size() > width)
        {
          sout << line << "\n";
          line.assign(column, ' ');
          first = true;
        }
        if (!first)
          line += ' ';
        line += word;
        first = false;
      }
      sout << line << "\n";
    }
  }

  sout <<
    "\n"
    "  --                          Treat all following arguments as filenames\n"
    "  @<file>                     Process a listing of files specified in a text\n"
    "                              file, or a bare @ to read it from stdin\n"
    "\n"
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

    sourceblockcount = 0;
    largestfilesize = 0;
    for (std::vector<std::string>::const_iterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
    {
      u64 filesize = filesize_cache.get(*i);
      sourceblockcount += (u32) ((filesize + blocksize-1) / blocksize);
      if (filesize > largestfilesize)
      {
	largestfilesize = filesize;
      }
    }

    if (sourceblockcount > 32768)
    {
      serr << "Too many source blocks (" << sourceblockcount << " > 32768)." << std::endl;
      return false;
    }

    if (!ComputeRecoveryBlockCount(serr,
				   &recoveryblockcount,
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
  std::string path, name;
  DiskFile::SplitFilename(argv[0], path, name);
  argc--;
  argv++;

  if (argc>0)
  {
    if (argv[0][0] == '-')
    {
      if (argv[0] == std::string("-h") || argv[0] == std::string("--help"))
      {
	usage();
	return true;
      }
      else if (argv[0] == std::string("-V") || argv[0] == std::string("--version"))
      {
	showversion();
	return true;
      }
      else if (argv[0] == std::string("-VV"))
      {
	showversion();
	sout << '\n';
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
      serr << "Not enough command line arguments." << std::endl;
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
      serr << "Invalid operation specified: " << argv[0] << std::endl;
      return false;
    }
    argc--;
    argv++;
  }

  basepath = "";

  cxxopts::Options parser(name);
  AddOptions(parser);

  // cxxopts skips the first argument, as the name the program was run as
  std::vector<const char *> args(1, name.c_str());
  args.insert(args.end(), argv, argv + argc);

  std::unique_ptr<cxxopts::ParseResult> parsed;
  try
  {
    parsed.reset(new cxxopts::ParseResult(parser.parse((int)args.size(), args.data())));
  }
  catch (const cxxopts::exceptions::exception &e)
  {
    serr << e.what() << std::endl;
    serr << "  (Run \"" << path << name << " --help\" for supported options.)" << std::endl;
    return false;
  }
  const cxxopts::ParseResult &result = *parsed;

  // The value of an option which takes one
  auto value = [&](const char *option) { return result[option].as<std::string>(); };

  u64 number = 0;

  if (result.count("help"))
  {
    usage();
    operation = opNone;
    return true;
  }

  if (result.count("version"))
  {
    showversion();
    if (result.count("version") > 1)
    {
      sout << '\n';
      banner();
    }
    operation = opNone;
    return true;
  }

  if (result.count("archive"))
  {
    if (!SetParFilename(value("archive")))
    {
      serr << "failed to set the main par file" << std::endl;
      return false;
    }
  }

  if (result.count("block-count"))  // Set the block count
  {
    if (operation != opCreate)
    {
      serr << "Cannot specify block count unless creating." << std::endl;
      return false;
    }
    if (result.count("block-count") > 1)
    {
      serr << "Cannot specify block count twice." << std::endl;
      return false;
    }
    else if (result.count("block-size"))
    {
      serr << "Cannot specify both block count and block size." << std::endl;
      return false;
    }
    if (!ParseNumber(value("block-count"), 32768, number) || 0 == number)
    {
      serr << "Invalid block count option: -b" << value("block-count") << std::endl;
      return false;
    }
    blockcount = (u32)number;
  }

  if (result.count("block-size"))  // Set the block size
  {
    if (operation != opCreate)
    {
      serr << "Cannot specify block size unless creating." << std::endl;
      return false;
    }
    if (result.count("block-size") > 1)
    {
      serr << "Cannot specify block size twice." << std::endl;
      return false;
    }
    if (!ParseNumber(value("block-size"), 0xFFFFFFFF, number) || 0 == number)
    {
      serr << "Invalid block size option: -s" << value("block-size") << std::endl;
      return false;
    }
    if (number & 3)
    {
      serr << "Block size must be a multiple of 4." << std::endl;
      return false;
    }
    blocksize = number;
  }

  if (result.count("threads"))  // Set amount of threads
  {
    if (!ParseNumber(value("threads"), 0xFFFFFFFF, number) || 0 == number)
    {
      serr << "Invalid thread option: -t" << value("threads") << std::endl;
      return false;
    }
    nthreads = (u32)std::min<u64>(number, MAX_THREAD_COUNT);
  }

  if (result.count("file-threads"))  // Set amount of file threads
  {
    if (!ParseNumber(value("file-threads"), 0xFFFFFFFF, number) || 0 == number)
    {
      serr << "Invalid file-thread option: -T" << value("file-threads") << std::endl;
      return false;
    }
    filethreads = (u32)number;
  }

  if (result.count("redundancy-size"))  // Set the size of the recovery data
  {
    if (operation != opCreate)
    {
      serr << "Cannot specify redundancy unless creating." << std::endl;
      return false;
    }
    if (result.count("redundancy-size") + result.count("redundancy") > 1)
    {
      serr << "Cannot specify redundancy twice." << std::endl;
      return false;
    }
    else if (result.count("recovery-blocks"))
    {
      serr << "Cannot specify both redundancy and recovery block count." << std::endl;
      return false;
    }

    const std::string sizevalue = value("redundancy-size");
    const char unit = sizevalue.empty() ? 0 : sizevalue[sizevalue.size() - 1];
    const int shift = unit == 'g' ? 30 : unit == 'm' ? 20 : unit == 'k' ? 10 : 0;
    if (!ParseNumber(shift ? sizevalue.substr(0, sizevalue.size() - 1) : sizevalue, ~(u64)0 >> shift, number) || 0 == number)
    {
      serr << "Invalid redundancy size option: --redundancy-size " << sizevalue << std::endl;
      return false;
    }
    redundancysize = number << shift;
    redundancyset = true;
  }

  if (result.count("redundancy"))  // Set the amount of redundancy required
  {
    if (operation != opCreate)
    {
      serr << "Cannot specify redundancy unless creating." << std::endl;
      return false;
    }
    if (result.count("redundancy") + result.count("redundancy-size") > 1)
    {
      serr << "Cannot specify redundancy twice." << std::endl;
      return false;
    }
    else if (result.count("recovery-blocks"))
    {
      serr << "Cannot specify both redundancy and recovery block count." << std::endl;
      return false;
    }

    const std::string redundancyvalue = value("redundancy");
    const char unit = redundancyvalue.empty() ? 0 : redundancyvalue[0];
    if (unit == 'k' || unit == 'm' || unit == 'g')
    {
      const int shift = unit == 'g' ? 30 : unit == 'm' ? 20 : 10;
      if (!ParseNumber(redundancyvalue.substr(1), ~(u64)0 >> shift, number))
      {
        serr << "Invalid redundancy option: -r" << redundancyvalue << std::endl;
        return false;
      }
      redundancysize = number << shift;
    }
    else
    {
      if (!ParseNumber(redundancyvalue, 0xFFFFFFFF, number))
      {
        serr << "Invalid redundancy option: -r" << redundancyvalue << std::endl;
        return false;
      }
      redundancy = (u32)number;
      if (redundancy > 100)
      {
        serr << "WARNING: Creating recovery file(s) with " << redundancy << "% redundancy." << std::endl;
      }
    }
    redundancyset = true;
  }

  if (result.count("recovery-blocks")) // Set the number of recovery blocks to create
  {
    if (operation != opCreate)
    {
      serr << "Cannot specify recovery block count unless creating." << std::endl;
      return false;
    }
    if (result.count("recovery-blocks") > 1)
    {
      serr << "Cannot specify recovery block count twice." << std::endl;
      return false;
    }
    if (!ParseNumber(value("recovery-blocks"), 32768, number))
    {
      serr << "Invalid recoveryblockcount option: -c" << value("recovery-blocks") << std::endl;
      return false;
    }
    recoveryblockcount = (u32)number;
    recoveryblockcountset = true;
  }

  if (result.count("first-recovery-block"))  // Specify the First block recovery number
  {
    if (operation != opCreate)
    {
      serr << "Cannot specify first block number unless creating." << std::endl;
      return false;
    }
    if (result.count("first-recovery-block") > 1)
    {
      serr << "Cannot specify first block twice." << std::endl;
      return false;
    }
    if (!ParseNumber(value("first-recovery-block"), 32768, number))
    {
      serr << "Invalid first block option: -f" << value("first-recovery-block") << std::endl;
      return false;
    }
    firstblock = (u32)number;
  }

  if (result.count("uniform"))  // Specify uniformly sized recovery files
  {
    if (operation != opCreate)
    {
      serr << "Cannot specify uniform files unless creating." << std::endl;
      return false;
    }
    recoveryfilescheme = scUniform;
  }

  if (result.count("limit"))  // Limit the size of the recovery files
  {
    if (operation != opCreate)
    {
      serr << "Cannot specify limit files unless creating." << std::endl;
      return false;
    }
    if (result.count("uniform"))
    {
      serr << "Cannot specify two recovery file size schemes." << std::endl;
      return false;
    }
    if (result.count("recovery-files"))
    {
      serr << "Cannot specify limited size and number of files at the same time." << std::endl;
      return false;
    }
    recoveryfilescheme = scLimited;
  }

  if (result.count("recovery-files"))  // Specify the number of recovery files
  {
    if (operation != opCreate)
    {
      serr << "Cannot specify recovery file count unless creating." << std::endl;
      return false;
    }
    if (result.count("recovery-files") > 1)
    {
      serr << "Cannot specify recovery file count twice." << std::endl;
      return false;
    }
    if (recoveryblockcountset && recoveryblockcount == 0)
    {
      serr << "Cannot set file count when recovery block count is set to 0." << std::endl;
      return false;
    }
    if (redundancyset && redundancy == 0 && redundancysize == 0)
    {
      serr << "Cannot set redundancy to 0 and file count > 0" << std::endl;
      return false;
    }
    if (!ParseNumber(value("recovery-files"), 0xFFFFFFFF, number) || 0 == number)
    {
      serr << "Invalid recovery file count option: -n" << value("recovery-files") << std::endl;
      return false;
    }

    // cap the maximum number of recovery files
    // because the number of recovery blocks will be
    // (2 ^ recoveryfilecount) - 1
    // the number 32 will overflow the u32 resulting in 1
    if (number > 31)
    {
      serr << "Invalid recovery file count option: " << number << std::endl;
      serr << "  the maximum allowed recovery file count is 31" << std::endl;
      return false;
    }
    recoveryfilecount = (u32)number;

    // When number of recovery files is used, set the recoveryfilescheme
    // to uniform, since variable will not always be able to fill all
    // the files
    recoveryfilescheme = scUniform;
  }

  if (result.count("memory"))  // Specify how much memory to use for output buffers
  {
    if (result.count("memory") > 1)
    {
      serr << "Cannot specify memory limit twice." << std::endl;
      return false;
    }
    if (!ParseNumber(value("memory"), SIZE_MAX / 1048576, number) || 0 == number)
    {
      serr << "Invalid memory limit option: -m" << value("memory") << std::endl;
      return false;
    }
    memorylimit = (size_t)number;
  }

  if (result.count("verbose") && result.count("quiet"))
  {
    serr << "Cannot use both -v and -q." << std::endl;
    return false;
  }
  if (result.count("verbose"))
    noiselevel = result.count("verbose") > 1 ? nlDebug : nlNoisy;
  if (result.count("quiet"))
    noiselevel = result.count("quiet") > 1 ? nlSilent : nlQuiet;

  if (result.count("purge"))
  {
    if (operation != opRepair && operation != opVerify)
    {
      serr << "Cannot specify purge unless repairing or verifying." << std::endl;
      return false;
    }
    purgefiles = true;
  }

  if (result.count("rename-only"))
  {
    if (operation != opRepair && operation != opVerify)
    {
      serr << "Cannot specify rename-only unless repairing or verifying." << std::endl;
      return false;
    }
    renameonly = true;
  }

  if (result.count("recurse"))
  {
    if (operation != opCreate)
    {
      serr << "Cannot specific Recursive unless creating." << std::endl;
      return false;
    }
    recursive = true;
  }

  if (result.count("follow-symlinks"))
  {
    if (operation != opCreate)
    {
      serr << "Cannot specify -L (follow symlinks) unless creating." << std::endl;
      return false;
    }
    followlinks = true;
  }

  if (result.count("data-skipping"))
  {
    if (operation == opCreate)
    {
      serr << "Cannot specify Data Skipping unless reparing or verifying." << std::endl;
      return false;
    }
    skipdata = true;
  }

  if (result.count("skip-leaway"))  // Set the skip leaway
  {
    if (operation == opCreate)
    {
      serr << "Cannot specify skip leaway when creating." << std::endl;
      return false;
    }
    if (!skipdata)
    {
      serr << "Cannot specify skip leaway and no skipping." << std::endl;
      return false;
    }
    if (!ParseNumber(value("skip-leaway"), 0xFFFFFFFF, number) || 0 == number)
    {
      serr << "Invalid skipleaway option: -S" << value("skip-leaway") << std::endl;
      return false;
    }
    skipleaway = number;
  }

  if (result.count("basepath")) // Set the basepath manually
  {
    basepath = DiskFile::GetCanonicalPathname(value("basepath"));
  }

  if (result.count("full-hash"))
  {
    if (operation == opCreate)
    {
      serr << "Cannot specify a full hash check unless repairing or verifying." << std::endl;
      return false;
    }
    fullhash = true;
  }

  // A name which matches no file is said, and the work goes on without it
  auto missing = [&](const std::string &filename, const std::string &name) {
    if (noiselevel == nlSilent)
      return;
    if (name.find_first_of("*?") != std::string::npos)
      sout << "No files match: " << filename << std::endl;
    else
      sout << "Ignoring non-existent source file: " << filename << std::endl;
  };

  // The PAR2 file, then the files, and the lists of them
  const std::vector<std::string> none;
  for (const std::string &arg : result.count("files") ? result["files"].as<std::vector<std::string> >() : none)
  {
    if (arg.empty())
      continue;

    if (arg[0] == '@') // Handle list files
    {
      std::istream* input = nullptr;
      std::ifstream listfile;

      // 1. Check if the '@' is followed by a filename
      if (arg.size() == 1)
      {
        // Bare '@' - read filelist from stdin
        input = &std::cin;
      }
      else
      {
        listfile.open(arg.substr(1));
        if (!listfile.is_open())
        {
          serr << "Could not open list file: " << arg.substr(1) << std::endl;
          return false;
        }
        input = &listfile;
      }

      std::string line;
      while (std::getline(*input, line))
      {
        // 2. Trim whitespace or skip whitespace-only lines
        // This finds the first non-whitespace character
        size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
          continue; // Line is empty or only whitespace

        // 3. All files in the list are treated as source files.

        std::string lpath, lname;
        DiskFile::SplitFilename(line, lpath, lname);
        std::unique_ptr< std::list<std::string> > filenames(
          DiskFile::FindFiles(lpath, lname, recursive, followlinks)
        );

        if (!filenames || filenames->empty())
          missing(line, lname);
        else
        {
          for (auto const& fn : *filenames)
          {
            rawfilenames.push_back(DiskFile::GetCanonicalPathname(fn));
          }
        }
      }
    }
    else if (parfilename.length() == 0)
    {
      if (!SetParFilename(arg))
      {
        serr << "failed to set the main par file" << std::endl;
        return false;
      }
    }
    else
    {
      std::string path;
      std::string name;
      DiskFile::SplitFilename(arg, path, name);
      std::unique_ptr< std::list<std::string> > filenames(
        DiskFile::FindFiles(path, name, recursive, followlinks)
      );

      if (!filenames || filenames->empty())
        missing(arg, name);
      else
      {
        for (auto const& fn : *filenames)
        {
          rawfilenames.push_back(DiskFile::GetCanonicalPathname(fn));
        }
      }
    }
  }

  return true;
}


bool CommandLine::CheckValuesAndSetDefaults() {
  if (parfilename.length() == 0)
  {
    serr << "You must specify a Recovery file." << std::endl;
    return false;
  }


  // Default noise level
  if (noiselevel == nlUnknown)
  {
    noiselevel = nlNormal;
  }

  // Default memorylimit of 256MB
  if (memorylimit == 0)
  {
    if (noiselevel >= nlDebug)
    {
      u64 TotalPhysicalMemory = GetTotalPhysicalMemory();

      if (TotalPhysicalMemory == 0)
        sout << "[DEBUG] could not detect physical memory" << std::endl;
      else
        sout << "[DEBUG] detected physical memory: " << TotalPhysicalMemory << " bytes" << std::endl;
    }

    memorylimit = DefaultMemoryLimit() / 1048576;
  }

  // limit to 1GB on 32-bit platforms to avoid exhausing the addressable memory space
  if (sizeof(uintptr_t) < 8 && memorylimit > 1024)
    memorylimit = 1024;

  // convert to megabytes
  memorylimit *= 1048576;

  if (noiselevel >= nlDebug)
  {
    sout << "[DEBUG] memorylimit: " << memorylimit << " bytes" << std::endl;
  }


  // Default basepath  (uses parfilename)
  if ("" == basepath)
  {
    if (noiselevel >= nlDebug)
    {
      sout << "[DEBUG] parfilename: " << parfilename << std::endl;
    }

    basepath = BasePathFor(parfilename);
  }

  basepath = WithSeparator(basepath);

  if (noiselevel >= nlDebug)
  {
    sout << "[DEBUG] basepath: " << basepath << std::endl;
  }


  // parfilename is checked earlier, because it is used by basepath.


  // check extrafiles
  std::list<std::string>::iterator rawfilenames_fn;
  for (rawfilenames_fn = rawfilenames.begin(); rawfilenames_fn != rawfilenames.end(); ++rawfilenames_fn)
  {
    std::string filename = *rawfilenames_fn;

    // Originally, all specified files were supposed to exist, or the program
    // would stop with an error message. This was not practical, for example in
    // a directory with files appearing and disappearing (an active download directory).
    // So the new rule is: when a specified file doesn't exist, it is silently skipped.
    if (!DiskFile::FileExists(filename))
    {
      sout << "Ignoring non-existent source file: " << filename << std::endl;
    }
    // skip files outside basepath
    else if (filename.compare(0, basepath.size(), basepath) != 0)
    {
      sout << "Ignoring out of basepath source file: " << filename << std::endl;
    }
    else
    {
      u64 filesize = filesize_cache.get(filename);

      // Ignore all 0 byte files
      if (filesize == 0)
      {
        sout << "Skipping 0 byte file: " << filename << std::endl;
      }
      else if (extrafiles.end() != std::find(extrafiles.begin(), extrafiles.end(), filename))
      {
        sout << "Skipping duplicate filename: " << filename << std::endl;
      }
      else
      {
        extrafiles.push_back(filename);
      }
    } //end file exists
  }


  // operation should always be set, but let's be thorough.
  if (operation == opNone) {
    serr << "ERROR: No operation was specified (create, repair, or verify)" << std::endl;
    return false;
  }


  if (operation != opCreate) {
    // skipdata is bool and either value is valid.

    // Default skip leaway
    if (skipdata && skipleaway == 0)
    {
      // Expect to find blocks within +/- 64 bytes of the expected
      // position relative to the last block that was found.
      skipleaway = DEFAULT_SKIP_LEAWAY;
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
        serr << "You must specify a list of files when creating." << std::endl;
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

          serr << "You must specify a list of files when creating." << std::endl;
          return false;
        }
      }
    }

    // Strip the ".par2" from the end of the filename of the main PAR2 file.
    parfilename = SetNameFor(parfilename);

    if (DiskFile::FileExists(parfilename + ".par2"))
    {
      serr << "Par2 file already exists: " << parfilename << std::endl;
      return false;
    }

    // If neither block count nor block size is specified, derive a default
    // block size from total source data, targeting ~2000 blocks (as previous
    // default). This replaces the old hard-coded blockcount=2000 which failed
    // when file count exceeded 2000, and avoids inflated block sizes when file
    // sizes vary widely. The minimum is clamped so that the total source block
    // count stays within the PAR2 limit of 32768.
    if (blockcount == 0 && blocksize == 0)
    {
        u64 totalsize = 0;
        for (auto &f : extrafiles)
            totalsize += filesize_cache.get(f);

        blocksize = (totalsize + 1999) / 2000; // 2000 = target (previous default)
        blocksize = std::max(blocksize, (u64)4);
        blocksize = (blocksize + 3) & ~3;

        u64 minblocksize = (totalsize + 32767) / 32768; // 32768 = maximum PAR2
        minblocksize = std::max(minblocksize, (u64)4);
        minblocksize = (minblocksize + 3) & ~3;

        if (blocksize < minblocksize)
            blocksize = minblocksize;
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
    std::vector<u64> filesizes;
    for (std::vector<std::string>::const_iterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
    {
      filesizes.push_back(filesize_cache.get(*i));
    }

    std::string error;
    if (!ComputeBlockSizeFromCount(&error, &blocksize, blockcount, filesizes))
    {
      serr << error << std::endl;
      return false;
    }
  }

  return true;
}


// Determine how many recovery blocks to create based on the source block
// count and the requested level of redundancy.
bool CommandLine::ComputeRecoveryBlockCount(std::ostream &serr,
					    u32 *recoveryblockcount,
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
    *recoveryblockcount = ComputeRecoveryBlockCountFromRedundancy(sourceblockcount, redundancy);
  }
  else if (redundancysize > 0)
  {
    const u64 overhead_per_recovery_file = sourceblockcount * (u64) 21;
    const u64 recovery_packet_size = blocksize + (u64) 70;
    if (recoveryfilecount == 0)
    {
      u32 estimatedFileCount = 15;
      u64 overhead = estimatedFileCount * overhead_per_recovery_file;
      u32 estimatedrecoveryblockcount;
      if (overhead > redundancysize)
      {
        estimatedrecoveryblockcount = 1;  // at least 1
      }
      else
      {
	estimatedrecoveryblockcount = (u32)((redundancysize - overhead) / recovery_packet_size);
      }

      // recoveryfilecount assigned below.
      std::string error;
      bool success = ComputeRecoveryFileCount(&error,
					      &recoveryfilecount,
					      recoveryfilescheme,
					      estimatedrecoveryblockcount,
					      largestfilesize,
					      blocksize);
      if (!success) {
	serr << error << std::endl;
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
    serr << "Redundancy and Redundancysize not set." << std::endl;
    return false;
  }

  if (*recoveryblockcount > 65536)
  {
    serr << "Too many recovery blocks requested." << std::endl;
    return false;
  }

  // Check that the last recovery block number would not be too large
  if (firstblock + *recoveryblockcount >= 65536)
  {
    serr << "First recovery block number is too high." << std::endl;
    return false;
  }

  return true;
}





bool CommandLine::SetParFilename(std::string filename)
{
  bool result = false;

#ifndef _WIN32
  std::string::size_type where;

  if ((where = filename.find_first_of('*')) != std::string::npos ||
      (where = filename.find_first_of('?')) != std::string::npos)
  {
    serr << "par2 file must not have a wildcard in it." << std::endl;
    return result;
  }
#endif

  // If we are verifying or repairing, the PAR2 file must
  // already exist
  if (operation != opCreate)
  {
    // Find the last '.' in the filename
    std::string::size_type where = filename.find_last_of('.');
    if (where != std::string::npos)
    {
      // Get what follows the last '.'
      std::string tail = filename.substr(where+1);

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

} // namespace par2
