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

#ifndef __PAR1REPAIRERSOURCEFILE_H__
#define __PAR1REPAIRERSOURCEFILE_H__

// The Par1RepairerSourceFile object is used during verification and repair
// to record details about a particular source file and the data blocks
// for that file.

class Par1RepairerSourceFile
{
public:
  // Construct the object and set the description and verification packets
  Par1RepairerSourceFile(PAR1FILEENTRY *fileentry, string searchpath);
  ~Par1RepairerSourceFile(void);

  string FileName(void) const {return filename;}
  u64 FileSize(void) const {return filesize;}
  const MD5Hash& HashFull(void) const {return hashfull;}
  const MD5Hash& Hash16k(void) const {return hash16k;}

  // Set/Get which DiskFile will contain the final repaired version of the file
  void SetTargetFile(DiskFile *diskfile);
  DiskFile* GetTargetFile(void) const;

  // Set/Get whether or not the target file actually exists
  void SetTargetExists(bool exists);
  bool GetTargetExists(void) const;

  // Set/Get which DiskFile contains a full undamaged version of the source file
  void SetCompleteFile(DiskFile *diskfile);
  DiskFile* GetCompleteFile(void) const;

  void SetTargetBlock(DiskFile *diskfile);

  DataBlock* SourceBlock(void) {return &sourceblock;}
  DataBlock* TargetBlock(void) {return &targetblock;}


protected:
  string       filename;
  u64          filesize;
  MD5Hash      hashfull;
  MD5Hash      hash16k;

  DataBlock    sourceblock;
  DataBlock    targetblock;

  bool         targetexists;        // Whether the target file exists
  DiskFile    *targetfile;          // The final version of the file
  DiskFile    *completefile;        // A complete version of the file


};


#endif // __PAR1REPAIRERSOURCEFILE_H__
