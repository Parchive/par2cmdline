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

static u32 smartpar11 = 0x03000101;

Par1Repairer::Par1Repairer(void)
{
  filelist = 0;
  filelistsize = 0;

  blocksize = 0;

  completefilecount = 0;
  renamedfilecount = 0;
  damagedfilecount = 0;
  missingfilecount = 0;

  inputbuffer = 0;
  outputbuffer = 0;

  noiselevel = CommandLine::nlNormal;
}

Par1Repairer::~Par1Repairer(void)
{
  delete [] (u8*)inputbuffer;
  delete [] (u8*)outputbuffer;

  map<u32,DataBlock*>::iterator i = recoveryblocks.begin();
  while (i != recoveryblocks.end())
  {
    DataBlock *datablock = i->second;
    delete datablock;

    ++i;
  }

  vector<Par1RepairerSourceFile*>::iterator sourceiterator = sourcefiles.begin();
  while (sourceiterator != sourcefiles.end())
  {
    Par1RepairerSourceFile *sourcefile = *sourceiterator;
    delete sourcefile;
    ++sourceiterator;
  }

  sourceiterator = extrafiles.begin();
  while (sourceiterator != extrafiles.end())
  {
    Par1RepairerSourceFile *sourcefile = *sourceiterator;
    delete sourcefile;
    ++sourceiterator;
  }

  delete [] filelist;
}

Result Par1Repairer::Process(const CommandLine &commandline, bool dorepair)
{
  // How noisy should we be
  noiselevel = commandline.GetNoiseLevel();

  // do we want to purge par files on success ?
  bool purgefiles = commandline.GetPurgeFiles();

  // Get filesnames from the command line
  string par1filename = commandline.GetParFilename();
  const list<CommandLine::ExtraFile> &extrafiles = commandline.GetExtraFiles();

  // Determine the searchpath from the location of the main PAR file
  string name;
  DiskFile::SplitFilename(par1filename, searchpath, name);

  // Load the main PAR file
  if (!LoadRecoveryFile(searchpath + name))
    return eLogicError;

  // Load other PAR files related to the main PAR file
  if (!LoadOtherRecoveryFiles(par1filename))
    return eLogicError;

  // Load any extra PAR files specified on the command line
  if (!LoadExtraRecoveryFiles(extrafiles))
    return eLogicError;

  if (noiselevel > CommandLine::nlQuiet)
    cout << endl << "Verifying source files:" << endl << endl;

  // Check for the existence of and verify each of the source files
  if (!VerifySourceFiles())
    return eFileIOError;

  if (completefilecount<sourcefiles.size())
  {
    if (noiselevel > CommandLine::nlQuiet)
      cout << endl << "Scanning extra files:" << endl << endl;

    // Check any other files specified on the command line to see if they are
    // actually copies of the source files that have the wrong filename
    if (!VerifyExtraFiles(extrafiles))
      return eLogicError;
  }

  // Find out how much data we have found
  UpdateVerificationResults();

  if (noiselevel > CommandLine::nlSilent)
    cout << endl;

  // Check the verification results and report the details
  if (!CheckVerificationResults())
    return eRepairNotPossible;

  // Are any of the files incomplete
  if (completefilecount<sourcefiles.size())
  {
    // Do we want to carry out a repair
    if (dorepair)
    {
      if (noiselevel > CommandLine::nlSilent)
        cout << endl;

      // Rename any damaged or missnamed target files.
      if (!RenameTargetFiles())
        return eFileIOError;

      // Are we still missing any files
      if (completefilecount<sourcefiles.size())
      {
        // Work out which files are being repaired, create them, and allocate
        // target DataBlocks to them, and remember them for later verification.
        if (!CreateTargetFiles())
          return eFileIOError;

        // Work out which data blocks are available, which need to be recreated, 
        // and compute the appropriate Reed Solomon matrix.
        if (!ComputeRSmatrix())
        {
          // Delete all of the partly reconstructed files
          DeleteIncompleteTargetFiles();
          return eFileIOError;
        }

        // Allocate memory buffers for reading and writing data to disk.
        if (!AllocateBuffers(commandline.GetMemoryLimit()))
        {
          // Delete all of the partly reconstructed files
          DeleteIncompleteTargetFiles();
          return eMemoryError;
        }
        if (noiselevel > CommandLine::nlSilent)
          cout << endl;

        // Set the total amount of data to be processed.
        progress = 0;
        totaldata = blocksize * sourcefiles.size() * verifylist.size();

        // Start at an offset of 0 within a block.
        u64 blockoffset = 0;
        while (blockoffset < blocksize) // Continue until the end of the block.
        {
          // Work out how much data to process this time.
          size_t blocklength = (size_t)min((u64)chunksize, blocksize-blockoffset);

          // Read source data, process it through the RS matrix and write it to disk.
          if (!ProcessData(blockoffset, blocklength))
          {
            // Delete all of the partly reconstructed files
            DeleteIncompleteTargetFiles();
            return eFileIOError;
          }

          // Advance to the need offset within each block
          blockoffset += blocklength;
        }

        if (noiselevel > CommandLine::nlSilent)
          cout << endl << "Verifying repaired files:" << endl << endl;

        // Verify that all of the reconstructed target files are now correct
        if (!VerifyTargetFiles())
        {
          // Delete all of the partly reconstructed files
          DeleteIncompleteTargetFiles();
          return eFileIOError;
        }
      }

      // Are all of the target files now complete?
      if (completefilecount<sourcefiles.size())
      {
        cerr << "Repair Failed." << endl;
        return eRepairFailed;
      }
      else
      {
        if (noiselevel > CommandLine::nlSilent)
          cout << endl << "Repair complete." << endl;
      }
    }
    else
    {
      return eRepairPossible;
    }
  }

  if (purgefiles == true)
  {
    RemoveBackupFiles();
    RemoveParFiles();
  }

  return eSuccess;
}

bool Par1Repairer::LoadRecoveryFile(string filename)
{
  // Skip the file if it has already been processed
  if (diskfilemap.Find(filename) != 0)
  {
    return true;
  }

  DiskFile *diskfile = new DiskFile;

  // Open the file
  if (!diskfile->Open(filename))
  {
    // If we could not open the file, ignore the error and 
    // proceed to the next file
    delete diskfile;
    return true;
  }

  if (noiselevel > CommandLine::nlSilent)
  {
    string path;
    string name;
    DiskFile::SplitFilename(filename, path, name);
    cout << "Loading \"" << name << "\"." << endl;
  }

  parlist.push_back(filename);

  bool havevolume = false;
  u32 volumenumber = 0;

  // How big is the file
  u64 filesize = diskfile->FileSize();
  if (filesize >= sizeof(PAR1FILEHEADER))
  {
    // Allocate a buffer to read data into
    size_t buffersize = (size_t)min((u64)1048576, filesize);
    u8 *buffer = new u8[buffersize];

    do
    {
      PAR1FILEHEADER fileheader;
      if (!diskfile->Read(0, &fileheader, sizeof(fileheader)))
        break;

      // Is this really a PAR file?
      if (fileheader.magic != par1_magic)
        break;

      // Is the version number correct?
      if (fileheader.fileversion != 0x00010000)
        break;

      ignore16kfilehash = (fileheader.programversion == smartpar11);

      // Prepare to carry out MD5 Hash check of the Control Hash
      MD5Context context;
      u64 offset = offsetof(PAR1FILEHEADER, sethash);

      // Process until the end of the file is reached
      while (offset < filesize)
      {
        // How much data should we read?
        size_t want = (size_t)min((u64)buffersize, filesize-offset);
        if (!diskfile->Read(offset, buffer, want))
          break;

        context.Update(buffer, want);

        offset += want;
      }

      // Did we read the whole file
      if (offset < filesize)
        break;

      // Compute the hash value
      MD5Hash hash;
      context.Final(hash);

      // Is it correct?
      if (hash != fileheader.controlhash)
        break;

      // Check that the volume number is ok
      if (fileheader.volumenumber >= 256)
        break;

      // Are there any files?
      if (fileheader.numberoffiles == 0 || 
          fileheader.filelistoffset < sizeof(PAR1FILEHEADER) ||
          fileheader.filelistsize == 0)
        break;

      // Verify that the file list and data offsets are ok
      if ((fileheader.filelistoffset + fileheader.filelistsize > filesize)
          ||
          (fileheader.datasize && (fileheader.dataoffset < sizeof(fileheader) || fileheader.dataoffset + fileheader.datasize > filesize))
          ||
          (fileheader.datasize && ((fileheader.filelistoffset <= fileheader.dataoffset && fileheader.dataoffset < fileheader.filelistoffset+fileheader.filelistsize) || (fileheader.dataoffset <= fileheader.filelistoffset && fileheader.filelistoffset < fileheader.dataoffset + fileheader.datasize))))
        break;

      // Check the size of the file list
      if (fileheader.filelistsize > 200000)
        break;

      // If we already have a copy of the file list, make sure this one has the same size
      if (filelist != 0 && filelistsize != fileheader.filelistsize)
        break;

      // Allocate a buffer to hold a copy of the file list
      unsigned char *temp = new unsigned char[(size_t)fileheader.filelistsize];

      // Read the file list into the buffer
      if (!diskfile->Read(fileheader.filelistoffset, temp, (size_t)fileheader.filelistsize))
      {
        delete [] temp;
        break;
      }

      // If we already have a copy of the file list, make sure this copy is identical
      if (filelist != 0)
      {
        bool match = (0 == memcmp(filelist, temp, filelistsize));
        delete [] temp;

        if (!match)
          break;
      }
      else
      {
        // Prepare to scan the file list
        unsigned char *current = temp;
        size_t remaining = (size_t)fileheader.filelistsize;
        unsigned int fileindex = 0;

        // Allocate a buffer to copy each file entry into so that
        // all fields will be correctly aligned in memory.
        PAR1FILEENTRY *fileentry = (PAR1FILEENTRY*)new u64[(remaining + sizeof(u64)-1)/sizeof(u64)];

        // Process until we run out of files or data
        while (remaining > 0 && fileindex < fileheader.numberoffiles)
        {
          // Copy fixed portion of file entry
          memcpy((void*)fileentry, (void*)current, sizeof(PAR1FILEENTRY));

          // Is there enough data remaining
          if (remaining < sizeof(fileentry->entrysize) ||
              remaining < fileentry->entrysize)
            break;

          // Check the length of the filename
          if (fileentry->entrysize <= sizeof(PAR1FILEENTRY))
            break;

          // Check the file size
          if (blocksize < fileentry->filesize)
            blocksize = fileentry->filesize;

          // Copy whole of file entry
          memcpy((void*)fileentry, (void*)current, (size_t)(u64)fileentry->entrysize);

          // Create source file and add it to the appropriate list
          Par1RepairerSourceFile *sourcefile = new Par1RepairerSourceFile(fileentry, searchpath);
          if (fileentry->status & INPARITYVOLUME)
          {
            sourcefiles.push_back(sourcefile);
          }
          else
          {
            extrafiles.push_back(sourcefile);
          }

          remaining -= (size_t)fileentry->entrysize;
          current += (size_t)fileentry->entrysize;

          fileindex++;
        }

        delete [] (u64*)fileentry;

        // Did we find the correct number of files
        if (fileindex < fileheader.numberoffiles)
        {
          vector<Par1RepairerSourceFile*>::iterator i = sourcefiles.begin();
          while (i != sourcefiles.end())
          {
            Par1RepairerSourceFile *sourcefile = *i;
            delete sourcefile;
            ++i;
          }
          sourcefiles.clear();

          i = extrafiles.begin();
          while (i != extrafiles.end())
          {
            Par1RepairerSourceFile *sourcefile = *i;
            delete sourcefile;
            ++i;
          }
          extrafiles.clear();

          delete [] temp;
          break;
        }

        filelist = temp;
        filelistsize = (u32)fileheader.filelistsize;
      }

      // Is this a recovery volume?
      if (fileheader.volumenumber > 0)
      {
        // Make sure there is data and that it is the correct size
        if (fileheader.dataoffset == 0 || fileheader.datasize != blocksize)
          break;

        // What volume number is this?
        volumenumber = (u32)(fileheader.volumenumber - 1);

        // Do we already have this volume?
        if (recoveryblocks.find(volumenumber) == recoveryblocks.end())
        {
          // Create a data block
          DataBlock *datablock = new DataBlock;
          datablock->SetLength(blocksize);
          datablock->SetLocation(diskfile, fileheader.dataoffset);

          // Store it in the map
          recoveryblocks.insert(pair<u32, DataBlock*>(volumenumber, datablock));

          havevolume = true;
        }
      }
    } while (false);

    delete [] buffer;
  }

  // We have finished with the file for now
  diskfile->Close();

  if (noiselevel > CommandLine::nlQuiet)
  {
    if (havevolume)
    {
      cout << "Loaded recovery volume " << volumenumber << endl;
    }
    else
    {
      cout << "No new recovery volumes found" << endl;
    }
  }

  // Remember that the file was processed
  bool success = diskfilemap.Insert(diskfile);
  assert(success);

  return true;
}

bool Par1Repairer::LoadOtherRecoveryFiles(string filename)
{
  // Split the original PAR filename into path and name parts
  string path;
  string name;
  DiskFile::SplitFilename(filename, path, name);

  // Find the file extension
  string::size_type where = name.find_last_of('.');
  if (where != string::npos)
  {
    // remove it
    name = name.substr(0, where);
  }

  // Search for additional PAR files
  string wildcard = name + ".???";
  list<string> *files = DiskFile::FindFiles(path, wildcard, false);

  for (list<string>::const_iterator s=files->begin(); s!=files->end(); ++s)
  {
    string filename = *s;

    // Find the file extension
    where = filename.find_last_of('.');
    if (where != string::npos)
    {
      string tail = filename.substr(where+1);

      // Check the file extension is the correct form
      if ((tail[0] == 'P' || tail[0] == 'p') &&
          (
            ((tail[1] == 'A' || tail[1] == 'a') && (tail[2] == 'R' || tail[2] == 'r'))
            ||
            (isdigit(tail[1]) && isdigit(tail[2]))
          ))
      {
        LoadRecoveryFile(filename);
      }
    }
  }

  delete files;

  return true;
}

// Load packets from any other PAR files whose names are given on the command line
bool Par1Repairer::LoadExtraRecoveryFiles(const list<CommandLine::ExtraFile> &extrafiles)
{
  for (ExtraFileIterator i=extrafiles.begin(); i!=extrafiles.end(); i++)
  {
    string filename = i->FileName();

    // Find the file extension
    string::size_type where = filename.find_last_of('.');
    if (where != string::npos)
    {
      string tail = filename.substr(where+1);

      // Check the file extension is the correct form
      if ((tail[0] == 'P' || tail[0] == 'p') &&
          (
            ((tail[1] == 'A' || tail[1] == 'a') && (tail[2] == 'R' || tail[2] == 'r'))
            ||
            (isdigit(tail[1]) && isdigit(tail[2]))
          ))
      {
        LoadRecoveryFile(filename);
      }
    }
  }

  return true;
}

// Attempt to verify all of the source files
bool Par1Repairer::VerifySourceFiles(void)
{
  bool finalresult = true;

  u32 filenumber = 0;
  vector<Par1RepairerSourceFile*>::iterator sourceiterator = sourcefiles.begin();
  while (sourceiterator != sourcefiles.end())
  {
    Par1RepairerSourceFile *sourcefile = *sourceiterator;

    string filename = sourcefile->FileName();

    // Check to see if we have already used this file
    if (diskfilemap.Find(filename) != 0)
    {
      string path;
      string name;
      DiskFile::SplitRelativeFilename(filename, path, name);

      // The file has already been used!
      cerr << "Source file " << name << " is a duplicate." << endl;

      finalresult = false;
    }
    else
    {
      DiskFile *diskfile = new DiskFile;

      // Does the target file exist
      if (diskfile->Open(filename))
      {
        // Yes. Record that fact.
        sourcefile->SetTargetExists(true);

        // Remember that the DiskFile is the target file
        sourcefile->SetTargetFile(diskfile);

        // Remember that we have processed this file
        bool success = diskfilemap.Insert(diskfile);
        assert(success);

        // Do the actual verification
        if (!VerifyDataFile(diskfile, sourcefile))
          finalresult = false;

        // We have finished with the file for now
        diskfile->Close();

        // Find out how much data we have found
        UpdateVerificationResults();
      }
      else
      {
        // The file does not exist.
        delete diskfile;

        if (noiselevel > CommandLine::nlSilent)
        {
          string path;
          string name;
          DiskFile::SplitFilename(filename, path, name);

          cout << "Target: \"" << name << "\" - missing." << endl;
        }
      }
    }

    ++sourceiterator;
    ++filenumber;
  }

  return finalresult;
}

// Scan any extra files specified on the command line
bool Par1Repairer::VerifyExtraFiles(const list<CommandLine::ExtraFile> &extrafiles)
{
  for (ExtraFileIterator i=extrafiles.begin(); 
       i!=extrafiles.end() && completefilecount<sourcefiles.size(); 
       ++i)
  {
    string filename = i->FileName();

    bool skip = false;

    // Find the file extension
    string::size_type where = filename.find_last_of('.');
    if (where != string::npos)
    {
      string tail = filename.substr(where+1);

      // Check the file extension is the correct form
      if ((tail[0] == 'P' || tail[0] == 'p') &&
          (
            ((tail[1] == 'A' || tail[1] == 'a') && (tail[2] == 'R' || tail[2] == 'r'))
            ||
            (isdigit(tail[1]) && isdigit(tail[2]))
          ))
      {
        skip = true;
      }
    }

    if (!skip)
    {
      filename = DiskFile::GetCanonicalPathname(filename);

      // Has this file already been dealt with
      if (diskfilemap.Find(filename) == 0)
      {
        DiskFile *diskfile = new DiskFile;

        // Does the file exist
        if (!diskfile->Open(filename))
        {
          delete diskfile;
          continue;
        }

        // Remember that we have processed this file
        bool success = diskfilemap.Insert(diskfile);
        assert(success);

        // Do the actual verification
        VerifyDataFile(diskfile, 0);
        // Ignore errors

        // We have finished with the file for now
        diskfile->Close();

        // Find out how much data we have found
        UpdateVerificationResults();
      }
    }
  }

  return true;
}


bool Par1Repairer::VerifyDataFile(DiskFile *diskfile, Par1RepairerSourceFile *sourcefile)
{
  Par1RepairerSourceFile *match = 0;

  string path;
  string name;
  DiskFile::SplitFilename(diskfile->FileName(), path, name);

  // How big is the file we are checking
  u64 filesize = diskfile->FileSize();

  if (filesize == 0)
  {
    if (noiselevel > CommandLine::nlSilent)
    {
      cout << "Target: \"" << name << "\" - empty." << endl;
    }
    return true;
  }

  // Search for the first file that is the correct size
  vector<Par1RepairerSourceFile*>::iterator sourceiterator = sourcefiles.begin();
  while (sourceiterator != sourcefiles.end() &&
         filesize != (*sourceiterator)->FileSize())
  {
    ++sourceiterator;
  }

  // Are there any files that are the correct size?
  if (sourceiterator != sourcefiles.end())
  {
    // Allocate a buffer to compute the file hash
    size_t buffersize = (size_t)min((u64)1048576, filesize);
    char *buffer = new char[buffersize];

    // Read the first 16k of the file
    size_t want = (size_t)min((u64)16384, filesize);
    if (!diskfile->Read(0, buffer, want))
    {
      delete [] buffer;
      return false;
    }

    // Compute the MD5 hash of the first 16k
    MD5Context contextfull;
    contextfull.Update(buffer, want);
    MD5Context context16k = contextfull;
    MD5Hash hash16k;
    context16k.Final(hash16k);

    if (!ignore16kfilehash)
    {
      // Search for the first file that has the correct 16k hash
      while (sourceiterator != sourcefiles.end() &&
            (filesize != (*sourceiterator)->FileSize() ||
              hash16k != (*sourceiterator)->Hash16k()))
      {
        ++sourceiterator;
      }
    }

    // Are there any files with the correct 16k hash?
    if (sourceiterator != sourcefiles.end())
    {
      // Compute the MD5 hash of the whole file
      if (filesize > 16384)
      {
        u64 progress = 0;
        u64 offset = 16384;
        while (offset < filesize)
        {
          if (noiselevel > CommandLine::nlQuiet)
          {
            // Update a progress indicator
            u32 oldfraction = (u32)(1000 * (progress) / filesize);
            u32 newfraction = (u32)(1000 * (progress=offset) / filesize);
            if (oldfraction != newfraction)
            {
              cout << "Scanning: \"" << name << "\": " << newfraction/10 << '.' << newfraction%10 << "%\r" << flush;
            }
          }

          want = (size_t)min((u64)buffersize, filesize-offset);

          if (!diskfile->Read(offset, buffer, want))
          {
            delete [] buffer;
            return false;
          }

          contextfull.Update(buffer, want);

          offset += want;
        }
      }

      MD5Hash hashfull;
      contextfull.Final(hashfull);

      // Search for the first file that has the correct full hash
      while (sourceiterator != sourcefiles.end() &&
            (filesize != (*sourceiterator)->FileSize() ||
              (!ignore16kfilehash && hash16k != (*sourceiterator)->Hash16k()) ||
              hashfull != (*sourceiterator)->HashFull()))
      {
        ++sourceiterator;
      }

      // Are there any files with the correct full hash?
      if (sourceiterator != sourcefiles.end())
      {
        // If a source file was originally specified, check to see if it is a match
        if (sourcefile != 0 &&
            sourcefile->FileSize() == filesize &&
            (ignore16kfilehash || sourcefile->Hash16k() == hash16k) &&
            sourcefile->HashFull() == hashfull)
        {
          match = sourcefile;
        }
        else
        {
          // Search for a file which matches and has not already been matched
          while (sourceiterator != sourcefiles.end() &&
                (filesize != (*sourceiterator)->FileSize() ||
                  (!ignore16kfilehash && hash16k != (*sourceiterator)->Hash16k()) ||
                  hashfull != (*sourceiterator)->HashFull() ||
                  (*sourceiterator)->GetCompleteFile() != 0))
          {
            ++sourceiterator;
          }

          // Did we find a match
          if (sourceiterator != sourcefiles.end())
          {
            match = *sourceiterator;
          }
        }
      }
    }

    delete [] buffer;
  }

  // Did we find a match
  if (match != 0)
  {
    match->SetCompleteFile(diskfile);

    if (noiselevel > CommandLine::nlSilent)
    {
      // Was the match the file we were originally looking for
      if (match == sourcefile)
      {
        cout << "Target: \"" << name << "\" - found." << endl;
      }
      // Were we looking for a specific file
      else if (sourcefile != 0)
      {
        string targetname;
        DiskFile::SplitFilename(sourcefile->FileName(), path, targetname);

        cout << "Target: \"" 
              << name 
              << "\" - is a match for \"" 
              << targetname 
              << "\"." 
              << endl;
      }
    }
    else
    {
      if (noiselevel > CommandLine::nlSilent)
      {
        string targetname;
        DiskFile::SplitFilename(match->FileName(), path, targetname);

        cout << "File: \"" 
              << name 
              << "\" - is a match for \"" 
              << targetname 
              << "\"." 
              << endl;
      }
    }
  }
  else
  {
    if (noiselevel > CommandLine:: nlSilent)
      cout << "File: \"" 
            << name 
            << "\" - no data found." 
            << endl;
  }

  return true;
}

void Par1Repairer::UpdateVerificationResults(void)
{
  completefilecount = 0;
  renamedfilecount = 0;
  damagedfilecount = 0;
  missingfilecount = 0;

  vector<Par1RepairerSourceFile*>::iterator sf = sourcefiles.begin();

  // Check the recoverable files
  while (sf != sourcefiles.end())
  {
    Par1RepairerSourceFile *sourcefile = *sf;

    // Was a perfect match for the file found
    if (sourcefile->GetCompleteFile() != 0)
    {
      // Is it the target file or a different one
      if (sourcefile->GetCompleteFile() == sourcefile->GetTargetFile())
      {
        completefilecount++;
      }
      else
      {
        renamedfilecount++;
      }
    }
    else
    {
      // Does the target file exist
      if (sourcefile->GetTargetExists())
      {
        damagedfilecount++;
      }
      else
      {
        missingfilecount++;
      }
    }

    ++sf;
  }
}

bool Par1Repairer::CheckVerificationResults(void)
{
  // Is repair needed
  if (completefilecount < sourcefiles.size() ||
      renamedfilecount > 0 ||
      damagedfilecount > 0 ||
      missingfilecount > 0)
  {
    if (noiselevel > CommandLine::nlSilent)
      cout << "Repair is required." << endl;
    if (noiselevel > CommandLine::nlQuiet)
    {
      if (renamedfilecount > 0) cout << renamedfilecount << " file(s) have the wrong name." << endl;
      if (missingfilecount > 0) cout << missingfilecount << " file(s) are missing." << endl;
      if (damagedfilecount > 0) cout << damagedfilecount << " file(s) exist but are damaged." << endl;
      if (completefilecount > 0) cout << completefilecount << " file(s) are ok." << endl;
    }

    // Is repair possible
    if (recoveryblocks.size() >= damagedfilecount+missingfilecount)
    {
      if (noiselevel > CommandLine::nlSilent)
        cout << "Repair is possible." << endl;

      if (noiselevel > CommandLine::nlQuiet)
      {
        if (recoveryblocks.size() > damagedfilecount+missingfilecount)
          cout << "You have an excess of " 
               << (u32)recoveryblocks.size() - (damagedfilecount+missingfilecount)
               << " recovery files." << endl;

        if (damagedfilecount+missingfilecount > 0)
          cout << damagedfilecount+missingfilecount
               << " recovery files will be used to repair." << endl;
        else if (recoveryblocks.size())
          cout << "None of the recovery files will be used for the repair." << endl;
      }

      return true;
    }
    else
    {
      if (noiselevel > CommandLine::nlSilent)
      {
        cout << "Repair is not possible." << endl;
        cout << "You need " << damagedfilecount+missingfilecount - recoveryblocks.size()
             << " more recovery files to be able to repair." << endl;
      }

      return false;
    }
  }
  else
  {
    if (noiselevel > CommandLine::nlSilent)
      cout << "All files are correct, repair is not required." << endl;

    return true;
  }

  return true;
}

bool Par1Repairer::RenameTargetFiles(void)
{
  vector<Par1RepairerSourceFile*>::iterator sf = sourcefiles.begin();

  // Rename any damaged target files
  while (sf != sourcefiles.end())
  {
    Par1RepairerSourceFile *sourcefile = *sf;

    // If the target file exists but is not a complete version of the file
    if (sourcefile->GetTargetExists() && 
        sourcefile->GetTargetFile() != sourcefile->GetCompleteFile())
    {
      DiskFile *targetfile = sourcefile->GetTargetFile();

      // Rename it
      diskfilemap.Remove(targetfile);

      if (!targetfile->Rename())
        return false;

      backuplist.push_back(targetfile);

      bool success = diskfilemap.Insert(targetfile);
      assert(success);

      // We no longer have a target file
      sourcefile->SetTargetExists(false);
      sourcefile->SetTargetFile(0);
    }

    ++sf;
  }

  sf = sourcefiles.begin();

  // Rename any missnamed but complete versions of the files
  while (sf != sourcefiles.end())
  {
    Par1RepairerSourceFile *sourcefile = *sf;

    // If there is no targetfile and there is a complete version
    if (sourcefile->GetTargetFile() == 0 &&
        sourcefile->GetCompleteFile() != 0)
    {
      DiskFile *targetfile = sourcefile->GetCompleteFile();

      // Rename it
      diskfilemap.Remove(targetfile);
      if (!targetfile->Rename(sourcefile->FileName()))
        return false;
      bool success = diskfilemap.Insert(targetfile);
      assert(success);

      // This file is now the target file
      sourcefile->SetTargetExists(true);
      sourcefile->SetTargetFile(targetfile);

      // We have one more complete file
      completefilecount++;
    }

    ++sf;
  }

  return true;
}

// Work out which files are being repaired, create them, and allocate
// target DataBlocks to them, and remember them for later verification.
bool Par1Repairer::CreateTargetFiles(void)
{
  vector<Par1RepairerSourceFile*>::iterator sf = sourcefiles.begin();

  // Create any missing target files
  while (sf != sourcefiles.end())
  {
    Par1RepairerSourceFile *sourcefile = *sf;

    // If the file does not exist
    if (!sourcefile->GetTargetExists())
    {
      DiskFile *targetfile = new DiskFile;
      string filename = sourcefile->FileName();
      u64 filesize = sourcefile->FileSize();

      // Create the target file
      if (!targetfile->Create(filename, filesize))
      {
        delete targetfile;
        return false;
      }

      // This file is now the target file
      sourcefile->SetTargetExists(true);
      sourcefile->SetTargetFile(targetfile);

      // Remember this file
      bool success = diskfilemap.Insert(targetfile);
      assert(success);

      sourcefile->SetTargetBlock(targetfile);

      // Add the file to the list of those that will need to be verified
      // once the repair has completed.
      verifylist.push_back(sourcefile);
    }

    ++sf;
  }

  return true;
}

// Work out which data blocks are available, which need to be recreated, 
// and compute the appropriate Reed Solomon matrix.
bool Par1Repairer::ComputeRSmatrix(void)
{
  inputblocks.resize(sourcefiles.size()); // The DataBlocks that will read from disk
  outputblocks.resize(verifylist.size()); // Those DataBlocks that will re recalculated

  vector<DataBlock*>::iterator inputblock  = inputblocks.begin();
  vector<DataBlock*>::iterator outputblock = outputblocks.begin();

  // Build an array listing which source data blocks are present and which are missing
  vector<bool> present;
  present.resize(sourcefiles.size());

  vector<Par1RepairerSourceFile*>::iterator sourceiterator = sourcefiles.begin();
  vector<bool>::iterator              pres = present.begin();

  // Iterate through all source files
  while (sourceiterator != sourcefiles.end())
  {
    Par1RepairerSourceFile *sourcefile = *sourceiterator;
    DataBlock *sourceblock = sourcefile->SourceBlock();
    DataBlock *targetblock = sourcefile->TargetBlock();

    // Was this block found
    if (sourceblock->IsSet())
    {
      // Open the file the block was found in.
      if (!sourceblock->Open())
      {
        return false;
      }

      // Record that the block was found
      *pres = true;

      // Add the block to the list of those which will be read 
      // as input (and which might also need to be copied).
      *inputblock = sourceblock;
      ++inputblock;
    }
    else
    {
      // Record that the block was missing
      *pres = false;

      // Add the block to the list of those to be written
      *outputblock = targetblock;
      ++outputblock;
    }

    ++sourceiterator;
    ++pres;
  }

  // Set the number of source blocks and which of them are present
  if (!rs.SetInput(present))
  {
    return false;
  }

  // Start iterating through the available recovery packets
  map<u32, DataBlock*>::iterator recoveryiterator = recoveryblocks.begin();

  // Continue to fill the remaining list of data blocks to be read
  while (inputblock != inputblocks.end())
  {
    // Get the next available recovery block
    u32        exponent      = recoveryiterator->first;
    DataBlock *recoveryblock = recoveryiterator->second;

    // Make sure the file is open
    if (!recoveryblock->Open())
    {
      return false;
    }
    // Add the recovery block to the list of blocks that will be read
    *inputblock = recoveryblock;

    // Record that the corresponding exponent value is the next one
    // to use in the RS matrix
    if (!rs.SetOutput(true, (u16)exponent))
    {
      return false;
    }

    ++inputblock;
    ++recoveryiterator;
  }

  // If we need to, compute and solve the RS matrix
  if (verifylist.size() == 0)
  {
    return true;
  }

  bool success = rs.Compute(noiselevel);
  return success;
}

// Allocate memory buffers for reading and writing data to disk.
bool Par1Repairer::AllocateBuffers(size_t memorylimit)
{
  // Would single pass processing use too much memory
  if (blocksize * verifylist.size() > memorylimit)
  {
    // Pick a size that is small enough
    chunksize = ~3 & (memorylimit / verifylist.size());
  }
  else
  {
    chunksize = (size_t)blocksize;
  }

  // Allocate the two buffers
  inputbuffersize = (size_t)chunksize;
  inputbuffer = new u8[inputbuffersize];
  outputbufferalignment = (inputbuffersize + sizeof(u32)-1) & ~(sizeof(u32)-1);
  outputbuffersize = outputbufferalignment * verifylist.size();
  outputbuffer = new u8[outputbuffersize];

  if (inputbuffer == NULL || outputbuffer == NULL)
  {
    cerr << "Could not allocate buffer memory." << endl;
    return false;
  }

  return true;
}

// Read source data, process it through the RS matrix and write it to disk.
bool Par1Repairer::ProcessData(u64 blockoffset, size_t blocklength)
{
  u64 totalwritten = 0;
  // Clear the output buffer
  memset(outputbuffer, 0, outputbuffersize);

  vector<DataBlock*>::iterator inputblock = inputblocks.begin();
  u32                          inputindex = 0;

  // Are there any blocks which need to be reconstructed
  if (verifylist.size() > 0)
  {
    // For each input block
    while (inputblock != inputblocks.end())       
    {
      // Read data from the current input block
      if (!(*inputblock)->ReadData(blockoffset, blocklength, inputbuffer))
        return false;

      // For each output block
      for (u32 outputindex=0; outputindex<verifylist.size(); outputindex++)
      {
        // Select the appropriate part of the output buffer
        void *outbuf = &outputbuffer[outputbufferalignment * outputindex];

        // Process the data
        rs.Process(blocklength, inputindex, inputbuffer, outputindex, outbuf);

        if (noiselevel > CommandLine::nlQuiet)
        {
          // Update a progress indicator
          u32 oldfraction = (u32)(1000 * progress / totaldata);
          progress += blocklength;
          u32 newfraction = (u32)(1000 * progress / totaldata);

          if (oldfraction != newfraction)
          {
            cout << "Repairing: " << newfraction/10 << '.' << newfraction%10 << "%\r" << flush;
          }
        }
      }

      ++inputblock;
      ++inputindex;
    }
  }

  if (noiselevel > CommandLine::nlQuiet)
    cout << "Writing recovered data\r";

  // For each output block that has been recomputed
  vector<DataBlock*>::iterator outputblock = outputblocks.begin();
  for (u32 outputindex=0; outputindex<verifylist.size();outputindex++)
  {
    // Select the appropriate part of the output buffer
    char *outbuf = &((char*)outputbuffer)[outputbufferalignment * outputindex];

    // Write the data to the target file
    size_t wrote;
    if (!(*outputblock)->WriteData(blockoffset, blocklength, outbuf, wrote))
      return false;
    totalwritten += wrote;

    ++outputblock;
  }

  if (noiselevel > CommandLine::nlQuiet)
    cout << "Wrote " << totalwritten << " bytes to disk" << endl;

  return true;
}

// Verify that all of the reconstructed target files are now correct
bool Par1Repairer::VerifyTargetFiles(void)
{
  bool finalresult = true;

  // Verify the target files in alphabetical order
//  sort(verifylist.begin(), verifylist.end(), SortSourceFilesByFileName);

  // Iterate through each file in the verification list
  for (list<Par1RepairerSourceFile*>::iterator sf = verifylist.begin();
       sf != verifylist.end();
       ++sf)
  {
    Par1RepairerSourceFile *sourcefile = *sf;
    DiskFile *targetfile = sourcefile->GetTargetFile();

    // Close the file
    if (targetfile->IsOpen())
      targetfile->Close();

    // Say we don't have a complete version of the file
    sourcefile->SetCompleteFile(0);

    // Re-open the target file
    if (!targetfile->Open())
    {
      finalresult = false;
      continue;
    }

    // Verify the file again
    if (!VerifyDataFile(targetfile, sourcefile))
      finalresult = false;

    // Close the file again
    targetfile->Close();

    // Find out how much data we have found
    UpdateVerificationResults();
  }

  return finalresult;
}

// Delete all of the partly reconstructed files
bool Par1Repairer::DeleteIncompleteTargetFiles(void)
{
  list<Par1RepairerSourceFile*>::iterator sf = verifylist.begin();

  // Iterate through each file in the verification list
  while (sf != verifylist.end())
  {
    Par1RepairerSourceFile *sourcefile = *sf;
    if (sourcefile->GetTargetExists())
    {
      DiskFile *targetfile = sourcefile->GetTargetFile();

      // Close and delete the file
      if (targetfile->IsOpen())
        targetfile->Close();
      targetfile->Delete();

      // Forget the file
      diskfilemap.Remove(targetfile);

      delete targetfile;

      // There is no target file
      sourcefile->SetTargetExists(false);
      sourcefile->SetTargetFile(0);
    }

    ++sf;
  }

  return true;
}

bool Par1Repairer::RemoveBackupFiles(void)
{
  vector<DiskFile*>::iterator bf = backuplist.begin();

  if (noiselevel > CommandLine::nlSilent
      && bf != backuplist.end())
  {
    cout << endl << "Purge backup files." << endl;
  }

  // Iterate through each file in the backuplist
  while (bf != backuplist.end())
  {
    if (noiselevel > CommandLine::nlSilent)
    {
      string name;
      string path;
      DiskFile::SplitFilename((*bf)->FileName(), path, name);
      cout << "Remove \"" << name << "\"." << endl;
    }

    if ((*bf)->IsOpen())
      (*bf)->Close();
    (*bf)->Delete();

    ++bf;
  }

  return true;
}

bool Par1Repairer::RemoveParFiles(void)
{
  if (noiselevel > CommandLine::nlSilent
      && parlist.size() > 0)
  {
      cout << endl << "Purge par files." << endl;
  }

  for (list<string>::const_iterator s=parlist.begin(); s!=parlist.end(); ++s)
  {
    DiskFile *diskfile = new DiskFile;

    if (diskfile->Open(*s))
    {
      if (noiselevel > CommandLine::nlSilent)
      {
        string name;
        string path;
        DiskFile::SplitFilename((*s), path, name);
        cout << "Remove \"" << name << "\"." << endl;
      }

      if (diskfile->IsOpen())
        diskfile->Close();
      diskfile->Delete();
    }

    delete diskfile;
  }

  return true;
}
