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

#ifndef __COMMANDLINE_H__
#define __COMMANDLINE_H__

// The CommandLine object is responsible for understanding the format
// of the command line parameters are parsing the command line to
// extract details as to what the user wants to do.

class CommandLine
{
public:
  CommandLine(void);

  // Parse the supplied command line arguments. 
  bool Parse(int argc, char *argv[]);

  // Display details of the correct format for command line parameters.
  static void usage(void);

  // What operation will we be carrying out
  typedef enum
  {
    opNone = 0,
    opCreate,        // Create new PAR2 recovery volumes
    opVerify,        // Verify but don't repair damaged data files
    opRepair         // Verify and if possible repair damaged data files
  } Operation;

  typedef enum
  {
    verUnknown = 0,
    verPar1,         // Processing PAR 1.0 files
    verPar2          // Processing PAR 2.0 files
  } Version;

  // Any extra files listed on the command line
  class ExtraFile
  {
  public:
    ExtraFile(void);
    ExtraFile(const ExtraFile&);
    ExtraFile& operator=(const ExtraFile&);

    ExtraFile(const string &name, u64 size);

    string FileName(void) const {return filename;}
    u64 FileSize(void) const {return filesize;}

  protected:
    string filename;
    u64    filesize;
  };

public:
  // Accessor functions for the command line parameters

  CommandLine::Operation GetOperation(void) const          {return operation;}
  CommandLine::Version   GetVersion(void) const            {return version;}
  u64                    GetBlockSize(void) const          {return blocksize;}
  u32                    GetBlockCount(void) const         {return blockcount;}
  u32                    GetRedundancy(void) const         {return redundancy;}
  u32                    GetFirstRecoveryBlock(void) const {return firstblock;}
  u32                    GetRecoveryFileCount(void) const  {return recoveryfilecount;}
  bool                   GetUniformFiles(void) const       {return uniformfiles;}
  size_t                 GetMemoryLimit(void) const        {return memorylimit;}

  string                              GetParFilename(void) const {return parfilename;}
  const list<CommandLine::ExtraFile>& GetExtraFiles(void) const  {return extrafiles;}

protected:
  Operation operation;         // The operation to be carried out.
  Version version;             // What version files will be processed.

  u32 blockcount;              // How many blocks the source files should 
                               // be virtually split into.

  u64 blocksize;               // What virtual block size to use.

  u32 firstblock;              // What the exponent value for the first
                               // recovery block will be.

  bool uniformfiles;           // Whether or not the recovery files should
                               // have an even distribution of recovery
                               // blocks, or an exponential one.

  u32 recoveryfilecount;       // How many recovery files should be created.

  u32 redundancy;              // What percentage of recovery data should
                               // be created.
  bool redundancyset;          // Set if the redunancy has be specified

  string parfilename;          // The name of the PAR2 file to create, or
                               // the name of the first PAR2 file to read
                               // when verifying or repairing.

  list<ExtraFile> extrafiles;  // The list of other files specified on the
                               // command line. When creating, this will be
                               // the source files, and when verifying or
                               // repairing, this will be additional PAR2
                               // files or data files to be examined.

  u64 totalsourcesize;         // Total size of the source files.

  u64 largestsourcesize;       // Size of the largest source file.

  size_t memorylimit;          // How much memory is permitted to be used
                               // for the output buffer when creating
                               // or repairing.
};

typedef list<CommandLine::ExtraFile>::const_iterator ExtraFileIterator;

#endif // __COMMANDLINE_H__
