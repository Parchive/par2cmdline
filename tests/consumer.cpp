//  This file is part of par2cmdline (a PAR 2.0 compatible file verification and
//  repair tool). See https://parchive.sourceforge.net for details of PAR 2.0.
//
//  Copyright (c) 2026 Michael Nightingale
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


// Built and run by tests/consumer_build with only the public include directory
// on the include path and without config.h, the way an embedding application
// sees libpar2. It creates its own recovery set, so it needs no fixtures.

#include <par2/libpar2.h>

#include <cstdio>
#include <algorithm>
#include <fstream>
#include <memory>
#include <mutex>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// An embedding application is free to use these names itself.
typedef double u32;
typedef char Result;

namespace
{
  const char *const DATA[] = {"consumer-0.data", "consumer-1.data", "consumer-2.data"};
  const size_t DATACOUNT = 3;
  const char *const PARFILE = "consumer.par2";
  const par2::u64 BLOCKSIZE = 4096;
  const par2::u32 RECOVERYBLOCKS = 20;

  int failures = 0;

  void Check(bool ok, const std::string &what)
  {
    if (ok)
      return;

    std::cerr << "FAILED: " << what << std::endl;
    ++failures;
  }

  // A Result which reports a failure always says why; one which reports an
  // outcome never does, however unwelcome the outcome is.
  void CheckLastError(const par2::Par2Verifier &verifier, const par2::Result result,
                      const std::string &what)
  {
    const bool failed = (result == par2::eInvalidCommandLineArguments ||
                         result == par2::eInsufficientCriticalData ||
                         result == par2::eFileIOError ||
                         result == par2::eLogicError ||
                         result == par2::eMemoryError);

    // A verify which reports damage may still have met a file it could not
    // read, which is worth keeping rather than a contradiction
    const bool damaged = (result == par2::eRepairPossible ||
                          result == par2::eRepairNotPossible ||
                          result == par2::eRepairFailed);

    par2::Par2Error error;
    error.code = par2::ecInternalError;

    const bool said = verifier.GetLastError(&error);

    Check(failed ? said : (damaged || !said), what + " reports an error only if it failed");
    Check(!said || error.code != par2::ecNone, what + " gives a code");
    Check(!said || !error.message.empty(), what + " gives a message");
  }

  // The same rule for a create, which reports no outcome of its own
  void CheckLastError(const par2::Par2Creator &creator, const par2::Result result,
                      const std::string &what)
  {
    const bool failed = (result != par2::eSuccess && result != par2::eCancelled);

    par2::Par2Error error;
    error.code = par2::ecInternalError;

    const bool said = creator.GetLastError(&error);

    Check(said == failed, what + " reports an error only if it failed");
    Check(!said || error.code != par2::ecNone, what + " gives a code");
    Check(!said || !error.message.empty(), what + " gives a message");
  }

  void WriteData(const char *name, unsigned seed, size_t bytes)
  {
    std::ofstream f(name, std::ios::binary | std::ios::trunc);
    for (size_t i = 0; i < bytes; ++i)
      f.put((char)((i * 31 + seed * 7) & 0xff));
  }

#ifndef _WIN32
  // Running as a user who may write to a read-only directory anyway would
  // prove nothing, so the check that relies on it asks first.
  bool CanWriteInto(const char *directory)
  {
    const std::string probe = std::string(directory) + "/probe";
    std::ofstream f(probe.c_str(), std::ios::binary | std::ios::trunc);
    const bool ok = f.good();
    f.close();
    std::remove(probe.c_str());
    return ok;
  }
#endif

  bool MakeDirectory(const char *name)
  {
#ifdef _WIN32
    return _mkdir(name) == 0;
#else
    return mkdir(name, 0755) == 0;
#endif
  }

  void Corrupt(const char *name, size_t offset, size_t bytes)
  {
    std::fstream f(name, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp((std::streamoff)offset);
    for (size_t i = 0; i < bytes; ++i)
      f.put((char)0x5a);
  }

  bool CreateSet(void)
  {
    std::vector<std::string> files;
    for (size_t i = 0; i < DATACOUNT; ++i)
    {
      WriteData(DATA[i], (unsigned)i, 20000 + i * 9000);
      files.emplace_back(DATA[i]);
    }

    std::ostringstream quiet;
    return par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                              64 * 1024 * 1024, "",
                                              0, 2,
                                              PARFILE, files,
                                              BLOCKSIZE, 0,
                                              par2::scVariable, 0, RECOVERYBLOCKS);
  }
}

// Counts what the observer is told, to show the callbacks arrive even when
// nothing is written to the output stream.
// Stops the work at the first sign of progress
class Canceller : public par2::Par2Observer
{
public:
  par2::Par2Verifier *verifier;
  Canceller() : verifier(0) {}
  void OnProgress(par2::Phase, par2::u32) override { if (verifier) verifier->Cancel(); }
};

// Stops a repair once it is reading back what it wrote
class LateCanceller : public par2::Par2Observer
{
public:
  par2::Par2Verifier *verifier;
  LateCanceller() : verifier(0) {}
  void OnProgress(par2::Phase phase, par2::u32) override
  {
    if (verifier && phase == par2::phVerifyingRepair)
      verifier->Cancel();
  }
};

class Counting : public par2::Par2Observer
{
public:
  Counting()
    : setinfo(0), files(0), progress(0), done(0), errors(0), warnings(0),
      lastinfo(), lasterror(), lastwarning(), phases(), last(0), wentbackwards(false),
      reached(false) {}

  int setinfo, files, progress, done, errors, warnings;

  // What the last OnSetInfo carried
  par2::Par2SetInfo lastinfo;

  // What the last OnError carried
  par2::Par2Error lasterror;

  // What the last OnWarning carried
  par2::Par2Warning lastwarning;

  // The steps reported, in order, with a run of the same step collapsed
  std::vector<par2::Phase> phases;

  // Enough to tell one run of progress from several
  par2::u32 last;
  bool wentbackwards, reached;

  void OnSetInfo(const par2::Par2SetInfo &info) override
  {
    std::lock_guard<std::mutex> lock(mutex);
    ++setinfo;
    lastinfo = info;
  }

  void OnFile(const std::string &) override
  {
    std::lock_guard<std::mutex> lock(mutex);
    ++files;
  }

  void OnFileDone(const std::string &, par2::u32, par2::u32) override
  {
    std::lock_guard<std::mutex> lock(mutex);
    ++done;
  }

  void OnError(const par2::Par2Error &error) override
  {
    std::lock_guard<std::mutex> lock(mutex);
    ++errors;
    lasterror = error;
  }

  void OnWarning(const par2::Par2Warning &warning) override
  {
    std::lock_guard<std::mutex> lock(mutex);
    ++warnings;
    lastwarning = warning;
  }

  void OnProgress(par2::Phase phase, par2::u32 permille) override
  {
    std::lock_guard<std::mutex> lock(mutex);

    ++progress;

    if (phases.empty() || phases.back() != phase)
    {
      phases.push_back(phase);
      last = 0;
    }

    if (permille < last)
      wentbackwards = true;
    if (permille == 1000)
      reached = true;
    last = permille;
  }

  bool Saw(par2::Phase phase) const
  {
    return std::find(phases.begin(), phases.end(), phase) != phases.end();
  }

private:
  // The callbacks arrive from the threads doing the work, several at a time
  std::mutex mutex;
};

int main()
{
  par2::u32 recoveryfilecount = 0;
  Check(par2::ComputeRecoveryFileCount(std::cout, std::cerr, &recoveryfilecount,
                                       par2::scVariable, 4, 1000, 100),
        "ComputeRecoveryFileCount");

  Check(CreateSet(), "par2create");

  std::ostringstream quiet;
  const std::vector<std::string> noextras;

  // A healthy set verifies, and the observer hears about it even at nlSilent
  {
    Counting observer;
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent);
    verifier.SetObserver(&observer);

    Check(par2::eSuccess == verifier.AddPar2File(PARFILE), "AddPar2File");
    CheckLastError(verifier, par2::eSuccess, "AddPar2File");

    par2::Par2SetInfo info;
    Check(verifier.GetSetInfo(&info), "GetSetInfo");
    Check(info.blocksize == BLOCKSIZE, "GetSetInfo blocksize");
    Check(info.recoverablefilecount == DATACOUNT, "GetSetInfo file count");
    bool setidset = false;
    for (par2::u8 byte : info.setid)
      setidset = setidset || byte != 0;
    Check(setidset, "GetSetInfo setid");
    Check(info.creator.find("Created by ") == 0, "GetSetInfo creator");
    // Counted from the packets, so it reads before anything has been verified
    Check(info.recoveryblocks == RECOVERYBLOCKS, "GetSetInfo recovery blocks");

    Check(observer.setinfo == 1, "OnSetInfo once");
    Check(observer.lastinfo.recoveryblocks == RECOVERYBLOCKS,
          "OnSetInfo recovery blocks");
    Check(observer.lastinfo.datablocks == info.datablocks, "OnSetInfo data blocks");
    Check(observer.lastinfo.blocksize == info.blocksize, "OnSetInfo blocksize");
    Check(observer.lastinfo.creator == info.creator, "OnSetInfo creator");

    std::vector<par2::Par2FileInfo> files;
    Check(verifier.GetFileInfo(&files), "GetFileInfo");
    Check(files.size() == DATACOUNT, "GetFileInfo count");
    for (const auto &file : files)
    {
      Check(file.hash16k.size() == 16, "GetFileInfo hash16k");
      Check(file.hashfull.size() == 16, "GetFileInfo hashfull");
      Check(file.hash16k != file.hashfull, "the two hashes differ");
    }

    par2::u32 totalblocks = 0;
    for (const auto &file : files)
    {
      Check(file.blockcount > 0, "GetFileInfo blockcount");
      totalblocks += file.blockcount;
    }
    Check(totalblocks == info.datablocks, "GetFileInfo blocks add up");

    for (const auto &file : files)
    {
      std::vector<par2::u32> crcs;
      Check(verifier.GetBlockChecksums(file.filename, &crcs),
            "GetBlockChecksums");
      Check(crcs.size() == file.blockcount,
            "GetBlockChecksums one entry per block");
    }
    {
      std::vector<par2::u32> crcs;
      Check(!verifier.GetBlockChecksums("not-in-the-set.bin", &crcs),
            "GetBlockChecksums rejects an unknown name");
      Check(crcs.empty(), "GetBlockChecksums clears on failure");
    }

    {
      std::vector<bool> found;
      Check(!verifier.GetFoundBlocks(files[0].filename, &found),
            "GetFoundBlocks says nothing before a verify");
    }

    Check(par2::eSuccess == verifier.Verify(noextras), "Verify healthy");

    par2::u32 foundblocks = 0;
    for (const auto &file : files)
    {
      std::vector<bool> found;
      Check(verifier.GetFoundBlocks(file.filename, &found), "GetFoundBlocks");
      Check(found.size() == file.blockcount,
            "GetFoundBlocks one entry per block");
      for (bool b : found)
        foundblocks += b ? 1 : 0;
    }
    {
      std::vector<bool> found;
      Check(!verifier.GetFoundBlocks("not-in-the-set.bin", &found),
            "GetFoundBlocks rejects an unknown name");
      Check(found.empty(), "GetFoundBlocks clears on failure");
    }

    par2::Par2VerifyResult status{};
    Check(verifier.GetVerifyResult(&status), "GetVerifyResult");
    Check(status.completefilecount == DATACOUNT, "all files complete");
    Check(status.missingblockcount == 0, "nothing missing");
    Check(status.availableblockcount == info.datablocks, "every block available");
    Check(foundblocks == status.availableblockcount,
          "an intact set holds every block that is available");
    Check(status.recoveryblockcount == info.recoveryblocks,
          "the verify counts the same recovery blocks");
    Check(quiet.str().empty(), "nlSilent writes nothing");
    Check(observer.setinfo == 1, "OnSetInfo called");
    Check(observer.progress > 0, "OnProgress called at nlSilent");
    // The PAR2 files read by AddPar2File are announced and finished too
    Check(observer.done == observer.files, "OnFileDone once per OnFile");
    Check(observer.done > (int)DATACOUNT, "the PAR2 files are counted as well");
  }

  // Two verifiers used one after the other do not disturb each other
  {
    par2::Par2Verifier a(quiet, quiet, par2::nlSilent);
    par2::Par2Verifier b(quiet, quiet, par2::nlSilent);

    Check(par2::eSuccess == a.AddPar2File(PARFILE), "first instance adds");
    par2::Par2SetInfo ainfo;
    Check(a.GetSetInfo(&ainfo), "first instance has the set");

    par2::Par2SetInfo binfo;
    Check(!b.GetSetInfo(&binfo), "second instance is still empty");

    Check(par2::eSuccess == b.AddPar2File(PARFILE), "second instance adds");
    Check(b.GetSetInfo(&binfo), "second instance has the set");
    Check(ainfo.setid == binfo.setid, "both see the same set");

    Check(par2::eSuccess == a.Verify(noextras), "first instance verifies");
    Check(par2::eSuccess == b.Verify(noextras), "second instance verifies");
  }

  // A basepath supplied at verify time has to reach the target names, which
  // are worked out when the packets are prepared
  {
    Check(MakeDirectory("elsewhere"), "mkdir");

    const char *const away[] = {"elsewhere/away-0.data", "elsewhere/away-1.data"};
    std::vector<std::string> files;
    for (size_t i = 0; i < 2; ++i)
    {
      WriteData(away[i], (unsigned)(i + 3), 30000);
      files.emplace_back(away[i]);
    }

    // created with a basepath, so the set records bare names
    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "elsewhere/", 0, 2,
                                             "elsewhere/away", files, BLOCKSIZE, 0,
                                             par2::scVariable, 0, 20),
          "par2create with a basepath");

    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "elsewhere/");
    Check(par2::eSuccess == verifier.AddPar2File("elsewhere/away.par2"), "AddPar2File");
    Check(par2::eSuccess == verifier.Verify(noextras),
          "Verify honours the verifier's basepath");

    // the names reported are the local names, with no basepath on them
    std::vector<par2::Par2FileInfo> reported;
    Check(verifier.GetFileInfo(&reported), "GetFileInfo");
    Check(reported.size() == 2, "GetFileInfo count");
    for (const auto &file : reported)
    {
      Check(file.filename.find('/') == std::string::npos, "no path in the name");
      Check(file.filename.find('\\') == std::string::npos, "no separator in the name");
    }

    // and known blocks are keyed by those same names
    for (const auto &file : reported)
      verifier.SetKnownBlocks(file.filename,
                              std::vector<bool>(file.blockcount, true));

    Check(par2::eSuccess == verifier.Verify(noextras),
          "known blocks are keyed by the reported name");

    for (const char *name : away)
      std::remove(name);
  }

  // Verifying more than once, and adding a PAR2 file after a verify, both work
  {
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent);

    Check(par2::eFileIOError == verifier.AddPar2File("no-such-file.par2"),
          "AddPar2File reports a name that is no use");

    Check(par2::eSuccess == verifier.AddPar2File(PARFILE), "AddPar2File");
    Check(par2::eSuccess == verifier.Verify(noextras), "first verify");
    Check(par2::eSuccess == verifier.Verify(noextras), "second verify");

    // naming a file of the set whose packets are already known is not an error
    Check(par2::eSuccess == verifier.AddPar2File(std::string(PARFILE) + ".par2"),
          "AddPar2File accepts an already known file of the set");

    // adding another of the set's files after verifying, then verifying again
    Check(par2::eSuccess == verifier.AddPar2File(PARFILE), "AddPar2File after verifying");
    Check(par2::eSuccess == verifier.Verify(noextras), "verify after adding");

    par2::Par2SetInfo info;
    Check(verifier.GetSetInfo(&info), "the set survives a restart");
    Check(info.recoverablefilecount == DATACOUNT, "file count survives a restart");
  }

  // Damage is found, and repairing puts it back
  {
    Corrupt(DATA[1], 5000, 2000);

    Counting observer;
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent);
    verifier.SetObserver(&observer);

    // the -m, -t and -T equivalents
    verifier.SetMemoryLimit(32 * 1048576);
    verifier.SetThreadCounts(2, 2);

    Check(par2::eSuccess == verifier.AddPar2File(PARFILE), "AddPar2File for damaged set");
    Check(par2::eRepairPossible == verifier.Verify(noextras),
          "Verify reports repair is possible");
    CheckLastError(verifier, par2::eRepairPossible, "a verify that found damage");

    // The set records its files in fileid order, not the order they were given
    std::vector<par2::Par2FileInfo> damagedfiles;
    verifier.GetFileInfo(&damagedfiles);
    size_t which = damagedfiles.size();
    for (size_t i = 0; i < damagedfiles.size(); ++i)
      if (damagedfiles[i].filename == DATA[1])
        which = i;
    Check(which < damagedfiles.size(), "the damaged file is in the set");

    std::vector<bool> damaged;
    Check(verifier.GetFoundBlocks(damagedfiles[which].filename, &damaged),
          "GetFoundBlocks for a damaged file");
    Check(damaged.size() == damagedfiles[which].blockcount,
          "GetFoundBlocks one entry per block of a damaged file");
    size_t missing = 0;
    for (bool b : damaged)
      missing += b ? 0 : 1;
    Check(missing > 0, "the damaged block is not found in the file");
    Check(missing < damaged.size(), "the undamaged ones still are");

    // The set's files are shifts of one periodic sequence, so the damaged
    // block turns up elsewhere and the repair does not have to rebuild it
    par2::Par2VerifyResult damagedstatus{};
    Check(verifier.GetVerifyResult(&damagedstatus), "GetVerifyResult for the damaged set");
    Check(damagedstatus.damagedfilecount == 1, "one file is damaged");

    // What one verifier found is what another may be told to take on trust
    {
      par2::Par2Verifier told(quiet, quiet, par2::nlSilent);
      Check(par2::eSuccess == told.AddPar2File(PARFILE), "AddPar2File for the vouched set");
      told.SetKnownBlocks(damagedfiles[which].filename, damaged);
      Check(par2::eRepairPossible == told.Verify(noextras),
            "the vouched blocks describe the same damage");

      std::vector<bool> again;
      Check(told.GetFoundBlocks(damagedfiles[which].filename, &again),
            "GetFoundBlocks after vouching");
      Check(again == damaged, "vouched blocks read back as found");
    }

    Check(par2::eSuccess == verifier.Repair(), "Repair");

    std::vector<bool> repaired;
    Check(verifier.GetFoundBlocks(damagedfiles[which].filename, &repaired),
          "GetFoundBlocks after a repair which read back what it wrote");
    size_t stillmissing = 0;
    for (bool b : repaired)
      stillmissing += b ? 0 : 1;
    Check(stillmissing == 0, "every block is found once the file is repaired");
  }

  // What the repair renamed out of the way can be tidied up
  {
    std::vector<std::string> leftovers;

    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent);
    Check(par2::eSuccess == verifier.AddPar2File(PARFILE), "AddPar2File");

    Corrupt(DATA[2], 2000, 3000);

    Check(par2::eRepairPossible == verifier.Verify(noextras),
          "Verify finds the damage");

    // nothing has been superseded yet
    Check(verifier.GetBackupFiles(&leftovers), "GetBackupFiles before repairing");
    Check(leftovers.empty(), "nothing has been renamed before a repair");

    Check(par2::eSuccess == verifier.Repair(), "Repair");

    // the damaged original was renamed out of the way, so it is now spare
    Check(verifier.GetBackupFiles(&leftovers), "GetBackupFiles after repairing");
    Check(leftovers.size() == 1, "the renamed original is reported");
    if (leftovers.size() == 1)
    {
      Check(leftovers[0].find(DATA[2]) != std::string::npos,
            "it is the file that was damaged");
      Check(leftovers[0] != DATA[2], "and not the repaired file itself");

      std::remove(leftovers[0].c_str());
    }
  }

  // and the repaired set verifies again
  {
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent);
    Check(par2::eSuccess == verifier.AddPar2File(PARFILE), "AddPar2File after repair");
    Check(par2::eSuccess == verifier.Verify(noextras), "Verify after repair");
  }

  // A memory limit of zero falls back to the default rather than stalling
  {
    Corrupt(DATA[2], 3000, 2000);

    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent);
    verifier.SetMemoryLimit(0);
    verifier.SetThreadCounts(0, 0);

    Check(par2::eSuccess == verifier.AddPar2File(PARFILE), "AddPar2File");
    Check(par2::eRepairPossible == verifier.Verify(noextras),
          "Verify finds the damage");
    Check(par2::eSuccess == verifier.Repair(), "Repair with a zero memory limit");
  }

  // Blocks the caller vouches for are not read, so a file whose contents are
  // wrong is still reported as intact
  {
    Corrupt(DATA[0], 1000, 8000);

    par2::Par2Verifier scanning(quiet, quiet, par2::nlSilent);
    Check(par2::eSuccess == scanning.AddPar2File(PARFILE), "AddPar2File before vouching");
    Check(par2::eRepairPossible == scanning.Verify(noextras),
          "damage is found when the file is scanned");

    par2::Par2Verifier trusting(quiet, quiet, par2::nlSilent);
    Check(par2::eSuccess == trusting.AddPar2File(PARFILE), "AddPar2File before vouching");

    std::vector<par2::Par2FileInfo> files;
    trusting.GetFileInfo(&files);
    for (const auto &file : files)
      trusting.SetKnownBlocks(file.filename,
                              std::vector<bool>(file.blockcount, true));

    Check(par2::eSuccess == trusting.Verify(noextras),
          "vouched blocks are taken on trust");

    // and being told a file holds nothing usable is also taken on trust
    par2::Par2Verifier writtenoff(quiet, quiet, par2::nlSilent);
    Check(par2::eSuccess == writtenoff.AddPar2File(PARFILE), "AddPar2File");

    std::vector<par2::Par2FileInfo> all;
    writtenoff.GetFileInfo(&all);
    writtenoff.SetKnownBlocks(all[0].filename, std::vector<bool>(all[0].blockcount, false));

    Check(par2::eRepairPossible == writtenoff.Verify(noextras),
          "a file written off is treated as unusable");

    par2::Par2VerifyResult writtenoffstatus{};
    Check(writtenoff.GetVerifyResult(&writtenoffstatus), "GetVerifyResult after writing a file off");
    Check(writtenoffstatus.missingblockcount >= all[0].blockcount,
          "its blocks are counted as missing");
  }

  // Cancelling stops the work and says so
  {
    Canceller canceller;
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent);

    Check(par2::eSuccess == verifier.AddPar2File(PARFILE), "AddPar2File before cancelling");

    // Attached once the packets are in, so the cancel lands in the scan
    canceller.verifier = &verifier;
    verifier.SetObserver(&canceller);

    Check(par2::eCancelled == verifier.Verify(noextras), "Verify is cancelled");
    CheckLastError(verifier, par2::eCancelled, "a cancelled verify");

    verifier.ClearCancel();
    verifier.SetObserver(0);

    // and verifying again starts the stopped pass afresh rather than carrying
    // on from where it was left
    Check(par2::eRepairPossible == verifier.Verify(noextras),
          "Verify again after ClearCancel finds the damage");

    par2::Par2Error error;
    Check(!verifier.GetLastError(&error) || error.code != par2::ecDuplicateSourceFile,
          "without taking the files the stopped pass had opened for duplicates");
  }

  // Reading the packets reports progress, so it can be cancelled too
  {
    Canceller canceller;
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent);
    canceller.verifier = &verifier;
    verifier.SetObserver(&canceller);

    Check(par2::eCancelled == verifier.AddPar2File(PARFILE),
          "AddPar2File is cancelled while it reads");
    CheckLastError(verifier, par2::eCancelled, "a cancelled AddPar2File");

    // What it read is not a set it can describe, and the cancelled name was
    // not remembered, so naming it again reads the rest
    verifier.ClearCancel();
    verifier.SetObserver(0);

    Check(par2::eSuccess == verifier.AddPar2File(PARFILE),
          "and naming it again after ClearCancel reads it properly");

    par2::Par2SetInfo info;
    Check(verifier.GetSetInfo(&info), "which leaves a set it can describe");
    Check(info.recoverablefilecount == DATACOUNT,
          "and every file of it, so the second read picked up where the cancel stopped");
  }

  // A cancel which lands in a volume found beside the named file leaves that
  // volume to be read again, rather than half read for good
  {
    Check(MakeDirectory("cutdir"), "mkdir for the cancelled volume check");

    const char *const data = "cutdir/cut.data";
    const char *const parfile = "cutdir/cut.par2";
    const char *const vol = "cutdir/cut.vol0+8.par2";

    WriteData(data, 23, 40000);

    std::vector<std::string> files;
    files.emplace_back(data);
    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "cutdir/", 0, 2,
                                             "cutdir/cut", files, BLOCKSIZE, 0,
                                             par2::scUniform, 1, 8),
          "par2create for the cancelled volume check");

    // Without the index file the volume is the first file read
    std::remove(parfile);

    Canceller canceller;
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "cutdir/");
    canceller.verifier = &verifier;
    verifier.SetObserver(&canceller);

    Check(par2::eCancelled == verifier.AddPar2File(parfile),
          "AddPar2File is cancelled while it reads the volume");

    verifier.ClearCancel();
    verifier.SetObserver(0);

    Check(par2::eSuccess == verifier.AddPar2File(parfile),
          "and naming the set again reads the rest of the volume");

    par2::Par2SetInfo info;
    Check(verifier.GetSetInfo(&info), "which leaves a set it can describe");
    Check(info.recoveryblocks == 8, "with every recovery block the volume holds");

    std::remove(data);
    std::remove(vol);
  }

  // Recovery data arriving a file at a time: what the scan found is kept, so
  // each new PAR2 file only needs a reassessment rather than another scan
  {
    const char *const data[] = {"stream-0.data", "stream-1.data"};
    const char *const vols[] = {"stream.vol00+3.par2", "stream.vol03+3.par2",
                                "stream.vol06+3.par2", "stream.vol09+3.par2",
                                "stream.vol12+3.par2", "stream.vol15+3.par2",
                                "stream.vol18+3.par2", "stream.vol21+3.par2"};
    const size_t volcount = sizeof(vols) / sizeof(vols[0]);

    std::vector<std::string> files;
    for (size_t i = 0; i < 2; ++i)
    {
      WriteData(data[i], (unsigned)(i + 9), 40000 + i * 5000);
      files.emplace_back(data[i]);
    }

    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "", 0, 2,
                                             "stream", files, BLOCKSIZE, 0,
                                             par2::scUniform, 8, 24),
          "par2create for the streaming set");

    // none of the recovery files have arrived yet
    for (const char *vol : vols)
      std::rename(vol, (std::string("held-") + vol).c_str());

    std::remove(data[1]);

    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent);
    Check(par2::eSuccess == verifier.AddPar2File("stream.par2"), "AddPar2File");
    Check(par2::eRepairNotPossible == verifier.Verify(noextras),
          "without recovery data a repair is not possible");

    Check(par2::eLogicError != verifier.Reassess(), "Reassess works after a verify");

    bool repaired = false;
    for (size_t i = 0; i < volcount && !repaired; ++i)
    {
      std::rename((std::string("held-") + vols[i]).c_str(), vols[i]);
      Check(par2::eSuccess == verifier.AddPar2File(vols[i]), "AddPar2File for a new volume");

      if (par2::eRepairPossible == verifier.Reassess())
      {
        Check(par2::eSuccess == verifier.Repair(), "Repair once enough blocks arrived");
        Check(par2::eSuccess == verifier.Verify(noextras),
              "the repaired set verifies");
        repaired = true;
      }
    }
    Check(repaired, "enough recovery data eventually made a repair possible");

    for (const char *name : data)
      std::remove(name);
  }

  // Reassess before anything has been verified says so
  {
    Counting observer;
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent);
    verifier.SetObserver(&observer);

    Check(par2::eLogicError == verifier.Reassess(), "Reassess needs a verify first");
    CheckLastError(verifier, par2::eLogicError, "Reassess without a verify");

    par2::Par2Error error;
    Check(verifier.GetLastError(&error), "Reassess without a verify says why");
    Check(error.code == par2::ecNotVerified, "and the reason is ecNotVerified");
    Check(observer.errors == 1, "the observer hears about it too");
    Check(observer.lasterror.code == par2::ecNotVerified, "with the same code");

    par2::Par2VerifyResult untouched;
    Check(!verifier.GetVerifyResult(&untouched), "GetVerifyResult needs a verify first");
    Check(untouched.completefilecount == 0 && untouched.missingblockcount == 0 &&
          untouched.recoveryblockcount == 0,
          "and a result it did not fill in reads as zero");
  }

  // A PAR2 file seen while it was still being written is read again when the
  // application names it, so the packets added since are not lost
  {
    // In a directory of its own, so the sibling search and the explicit add
    // name the file the same way. With both in the working directory the
    // search records a different spelling, the skip misses, and the file is
    // read again by accident rather than by intent.
    Check(MakeDirectory("partialdir"), "mkdir for the partial-file check");

    const char *const data = "partialdir/partial.data";
    const char *const setname = "partialdir/partial";
    const char *const parfile = "partialdir/partial.par2";
    const char *const vol = "partialdir/partial.vol0+8.par2";

    WriteData(data, 21, 40000);

    std::vector<std::string> files;
    files.emplace_back(data);
    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "partialdir/", 0, 2,
                                             setname, files, BLOCKSIZE, 0,
                                             par2::scUniform, 1, 8),
          "par2create for the partial-file check");

    // Keep a whole copy, then leave only the first half in place, as though
    // the volume file were still downloading when the sibling search ran.
    std::ifstream whole(vol, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(whole)),
                      std::istreambuf_iterator<char>());
    whole.close();

    std::ofstream half(vol, std::ios::binary | std::ios::trunc);
    half.write(bytes.data(), (std::streamsize)(bytes.size() / 2));
    half.close();

    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "partialdir/");
    Check(par2::eSuccess == verifier.AddPar2File(parfile),
          "AddPar2File finds the half-written volume");
    Check(par2::eSuccess == verifier.Verify(noextras),
          "the data itself is intact");

    par2::Par2VerifyResult partial{};
    Check(verifier.GetVerifyResult(&partial), "GetVerifyResult for the partial volume");

    // The rest of the volume arrives
    std::ofstream rest(vol, std::ios::binary | std::ios::trunc);
    rest.write(bytes.data(), (std::streamsize)bytes.size());
    rest.close();

    Check(par2::eSuccess == verifier.AddPar2File(vol), "AddPar2File for the completed volume");
    Check(par2::eLogicError != verifier.Reassess(), "Reassess after the volume completed");

    par2::Par2VerifyResult complete{};
    Check(verifier.GetVerifyResult(&complete), "GetVerifyResult after completion");
    Check(complete.recoveryblockcount > partial.recoveryblockcount,
          "the packets written since are picked up");

    std::remove(data);
    std::remove(parfile);
    std::remove(vol);
  }

  // A set which learns of more files after a cancelled verify is started
  // afresh, so the next verify looks for the files it did not know about
  {
    Check(MakeDirectory("growdir"), "mkdir for the growing set check");

    const char *const data[] = {"growdir/grow-0.data", "growdir/grow-1.data"};
    const char *const parfile = "growdir/grow.par2";
    const char *const vol = "growdir/grow.vol0+8.par2";
    const char *const held = "growdir/held.bin";

    std::vector<std::string> files;
    for (size_t i = 0; i < 2; ++i)
    {
      WriteData(data[i], 31 + (unsigned)i, 40000);
      files.emplace_back(data[i]);
    }

    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "growdir/", 0, 2,
                                             "growdir/grow", files, BLOCKSIZE, 0,
                                             par2::scUniform, 1, 8),
          "par2create for the growing set check");

    // An index holding the main packet and what describes only the first file.
    // Each packet stands on its own, so a file of some of them is still valid.
    std::ifstream whole(parfile, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(whole)),
                      std::istreambuf_iterator<char>());
    whole.close();

    std::string partial;
    std::string firstfile;
    for (size_t offset = 0; offset + 64 <= bytes.size();)
    {
      par2::u64 length = 0;
      for (size_t b = 0; b < 8; ++b)
        length |= (par2::u64)(unsigned char)bytes[offset + 8 + b] << (8 * b);

      const std::string type = bytes.substr(offset + 48, 16);
      const std::string packet = bytes.substr(offset, (size_t)length);
      const std::string fileid = length >= 80 ? packet.substr(64, 16) : std::string();

      if (type.compare(0, 12, std::string("PAR 2.0\0Main", 12)) == 0)
        partial += packet;
      else if (type.compare(0, 16, std::string("PAR 2.0\0FileDesc", 16)) == 0 && firstfile.empty())
      {
        firstfile = fileid;
        partial += packet;
      }
      else if (type.compare(0, 12, std::string("PAR 2.0\0IFSC", 12)) == 0 && fileid == firstfile)
        partial += packet;

      offset += (size_t)length;
    }

    std::ofstream cut(parfile, std::ios::binary | std::ios::trunc);
    cut.write(partial.data(), (std::streamsize)partial.size());
    cut.close();

    // Out of the way of the search for volumes beside the index
    std::rename(vol, held);

    Canceller canceller;
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "growdir/");
    Check(par2::eSuccess == verifier.AddPar2File(parfile),
          "AddPar2File for the index describing one file");

    std::vector<par2::Par2FileInfo> known;
    Check(verifier.GetFileInfo(&known) && known.size() == 1, "only the first file is known");

    canceller.verifier = &verifier;
    verifier.SetObserver(&canceller);

    Check(par2::eCancelled == verifier.Verify(noextras), "the verify is cancelled");

    verifier.ClearCancel();
    verifier.SetObserver(0);

    std::rename(held, vol);

    Check(par2::eSuccess == verifier.AddPar2File(vol), "AddPar2File for the volume");
    Check(verifier.GetFileInfo(&known) && known.size() == 2, "which describes both files");
    Check(par2::eSuccess == verifier.Verify(noextras), "and both are found intact");

    par2::Par2VerifyResult grown{};
    Check(verifier.GetVerifyResult(&grown) && grown.completefilecount == 2,
          "counting both of them");

    for (const char *name : data)
      std::remove(name);
    std::remove(parfile);
    std::remove(vol);
  }

  // What the observer is told: one run of progress per operation, files that
  // pair with their results, and PAR2 files reported separately
  {
    Check(MakeDirectory("observedir"), "mkdir for the observer check");

    // Several files, because with only one a per-file meter and a per-operation
    // meter cover the same bytes and the difference between them cannot be
    // seen. Big enough that a scan reports more than once.
    const char *const observed[] = {"observedir/observed-0.data",
                                    "observedir/observed-1.data",
                                    "observedir/observed-2.data"};
    const size_t observedcount = sizeof(observed) / sizeof(observed[0]);

    std::vector<std::string> files;
    for (size_t i = 0; i < observedcount; ++i)
    {
      WriteData(observed[i], (unsigned)(71 + i), 2000000);
      files.emplace_back(observed[i]);
    }
    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "observedir/", 0, 2,
                                             "observedir/observed", files, 20000, 0,
                                             par2::scUniform, 1, 20),
          "par2create for the observer check");

    Counting observer;
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "observedir/");
    verifier.SetObserver(&observer);

    Check(par2::eSuccess == verifier.AddPar2File("observedir/observed.par2"),
          "AddPar2File for the observer check");

    // PAR2 files are announced too, and each is closed off the same way
    Check(observer.files > 0, "OnFile called while reading PAR2 files");
    Check(observer.files == observer.done, "and each one is finished");
    Check(observer.progress > 0, "AddPar2File reports progress as it reads");
    Check(observer.reached, "and each file it read was finished");

    // Each file read is a run of its own, so the count starts again at each
    // one. Only the scan below is a single run.
    observer.last = 0;
    observer.reached = false;
    observer.wentbackwards = false;

    const int par2files = observer.files;

    Check(par2::eSuccess == verifier.Verify(noextras), "Verify for the observer check");

    Check(observer.progress > 1, "a scan reports progress more than once");
    Check(!observer.wentbackwards, "and it only ever goes up");
    Check(observer.reached, "reaching the end");
    Check(observer.files == observer.done, "every file reported is a file finished");
    Check(observer.files - par2files == (int)observedcount,
          "one report per file in the set, on top of the PAR2 files");

    // Reading the packets and scanning the data are told apart by the phase,
    // in the order they happened
    Check(observer.phases.size() == 2, "two steps were reported");
    Check(observer.phases.front() == par2::phLoading, "the packets were read first");
    Check(observer.phases.back() == par2::phScanning, "and the data scanned after");

    // A file the set describes but which is not on disk is announced and
    // finished like any other
    std::remove(observed[0]);

    Counting gone;
    par2::Par2Verifier missing(quiet, quiet, par2::nlSilent, "observedir/");
    missing.SetObserver(&gone);

    Check(par2::eSuccess == missing.AddPar2File("observedir/observed.par2"),
          "AddPar2File for the missing file check");

    const int gonepar2files = gone.files;

    Check(par2::eRepairNotPossible == missing.Verify(noextras),
          "Verify with one file of the set missing");
    Check(gone.files == gone.done, "the file which is not there is finished too");
    Check(gone.files - gonepar2files == (int)observedcount,
          "and is still one report per file in the set");

    for (size_t i = 1; i < observedcount; ++i)
      std::remove(observed[i]);
  }

  // A repair can be asked not to read back what it wrote
  {
    Check(MakeDirectory("skipdir"), "mkdir for the skipped-verification check");

    const char *const data = "skipdir/skip.data";
    WriteData(data, 61, 30000);

    std::vector<std::string> files;
    files.emplace_back(data);
    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "skipdir/", 0, 2,
                                             "skipdir/skip", files, BLOCKSIZE, 0,
                                             par2::scUniform, 1, 8),
          "par2create for the skipped-verification check");

    Corrupt(data, 500, 400);

    Counting observer;
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "skipdir/");
    verifier.SetObserver(&observer);

    Check(par2::eSuccess == verifier.AddPar2File("skipdir/skip.par2"),
          "AddPar2File for the skipped-verification check");
    Check(par2::eRepairPossible == verifier.Verify(noextras),
          "the damage is repairable");

    Check(par2::eSuccess == verifier.Repair(false), "Repair without reading it back");

    // Whether the read-back happened is a step of its own, so an application
    // can see that it was skipped rather than infer it
    Check(observer.Saw(par2::phScanning), "the data was scanned");
    Check(observer.Saw(par2::phProcessing), "and the repair written");
    Check(!observer.Saw(par2::phVerifyingRepair),
          "but nothing was read back, which is what was asked for");

    // Nothing recounted the files, so the numbers still describe the damage
    par2::Par2VerifyResult stale{};
    Check(verifier.GetVerifyResult(&stale), "GetVerifyResult after skipping");
    Check(stale.damagedfilecount == 1,
          "the counts still describe the state before the repair");

    // The repair itself was real, which a fresh verifier can say
    par2::Par2Verifier after(quiet, quiet, par2::nlSilent, "skipdir/");

    Check(par2::eSuccess == after.AddPar2File("skipdir/skip.par2"),
          "AddPar2File to check the repair");
    Check(par2::eSuccess == after.Verify(noextras),
          "the file really was repaired");

    std::remove(data);
  }

  // A cancel once every block has been written only stops the checking, and
  // the files the repair rebuilt are kept
  {
    Check(MakeDirectory("latedir"), "mkdir for the late cancel check");

    const char *const data = "latedir/late.data";
    const char *const backup = "latedir/late.data.1";
    const char *const parfile = "latedir/late.par2";
    const char *const vol = "latedir/late.vol0+8.par2";

    WriteData(data, 25, 40000);

    std::vector<std::string> files;
    files.emplace_back(data);
    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "latedir/", 0, 2,
                                             "latedir/late", files, BLOCKSIZE, 0,
                                             par2::scUniform, 1, 8),
          "par2create for the late cancel check");

    Corrupt(data, 1000, 5000);

    LateCanceller canceller;
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "latedir/");
    Check(par2::eSuccess == verifier.AddPar2File(parfile), "AddPar2File for the late cancel check");
    Check(par2::eRepairPossible == verifier.Verify(noextras), "the damage is found");

    canceller.verifier = &verifier;
    verifier.SetObserver(&canceller);

    Check(par2::eCancelled == verifier.Repair(),
          "a cancel while the repair reads back is still a cancel");

    par2::Par2Verifier after(quiet, quiet, par2::nlSilent, "latedir/");
    Check(par2::eSuccess == after.AddPar2File(parfile), "AddPar2File after the late cancel");
    Check(par2::eSuccess == after.Verify(noextras), "and the rebuilt file was kept, whole");

    std::remove(data);
    std::remove(backup);
    std::remove(parfile);
    std::remove(vol);
  }

  // A file found under another name is reported as a pair, and stays reported
  // after the repair that renames it into place
  {
    Check(MakeDirectory("renamedir"), "mkdir for the rename check");

    const char *const data = "renamedir/proper.data";
    const char *const obfuscated = "renamedir/9f3ac1b7e2.dat";

    WriteData(data, 51, 12000);

    std::vector<std::string> files;
    files.emplace_back(data);
    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "renamedir/", 0, 2,
                                             "renamedir/rename", files, BLOCKSIZE, 0,
                                             par2::scUniform, 1, 4),
          "par2create for the rename check");

    std::rename(data, obfuscated);

    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "renamedir/");
    Check(par2::eSuccess == verifier.AddPar2File("renamedir/rename.par2"),
          "AddPar2File for the rename check");

    std::vector<std::pair<std::string, std::string> > renamed;
    Check(verifier.GetRenamedFiles(&renamed), "GetRenamedFiles before verifying");
    Check(renamed.empty(), "nothing is renamed until something has been verified");

    std::vector<std::string> extras;
    extras.emplace_back(obfuscated);
    Check(par2::eRepairPossible == verifier.Verify(extras),
          "the file is found under its other name");

    Check(verifier.GetRenamedFiles(&renamed), "GetRenamedFiles after verifying");
    Check(renamed.size() == 1, "one file was found renamed");
    if (renamed.size() == 1)
    {
      const std::string had = renamed[0].first;
      const std::string belongs = renamed[0].second;

      Check(had.size() >= std::string("9f3ac1b7e2.dat").size() &&
            had.compare(had.size() - std::string("9f3ac1b7e2.dat").size(),
                        std::string::npos, "9f3ac1b7e2.dat") == 0,
            "reported under the name it had");
      Check(belongs.size() >= std::string("proper.data").size() &&
            belongs.compare(belongs.size() - std::string("proper.data").size(),
                            std::string::npos, "proper.data") == 0,
            "paired with the name it belongs under");

      // The point of canonicalising: the two halves are in one form, so the
      // directory they name compares equal rather than merely resolving alike
      Check(had.substr(0, had.find_last_of("/\\")) ==
            belongs.substr(0, belongs.find_last_of("/\\")),
            "both halves name the directory the same way");

      std::ifstream reachable(had.c_str(), std::ios::binary);
      Check(reachable.good(), "and that name opens the file");
    }

    Check(par2::eSuccess == verifier.Repair(), "Repair applies the rename");

    // The rename has happened, so nothing on disk shows it any more - the
    // report has to survive on its own
    std::vector<std::pair<std::string, std::string> > afterwards;
    Check(verifier.GetRenamedFiles(&afterwards), "GetRenamedFiles after repairing");
    Check(afterwards == renamed, "and it still says the same thing");

    std::ifstream restored(data, std::ios::binary);
    Check(restored.good(), "the file is under its proper name now");

    std::remove(data);
  }

  // A file scanned again once it no longer holds the set's data stops being
  // reported as that file under another name
  {
    Check(MakeDirectory("restaledir"), "mkdir for the rescanned rename check");

    const char *const data = "restaledir/proper.data";
    const char *const obfuscated = "restaledir/c4d2e19a.dat";

    WriteData(data, 53, 12000);

    std::vector<std::string> files;
    files.emplace_back(data);
    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "restaledir/", 0, 2,
                                             "restaledir/stale", files, BLOCKSIZE, 0,
                                             par2::scUniform, 1, 4),
          "par2create for the rescanned rename check");

    std::rename(data, obfuscated);

    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "restaledir/");
    Check(par2::eSuccess == verifier.AddPar2File("restaledir/stale.par2"),
          "AddPar2File for the rescanned rename check");
    Check(par2::eRepairPossible == verifier.VerifyFile(obfuscated),
          "the file is found under its other name");

    std::vector<std::pair<std::string, std::string> > renamed;
    Check(verifier.GetRenamedFiles(&renamed) && renamed.size() == 1,
          "and reported as renamed");

    // Its contents change, and it is scanned again
    WriteData(obfuscated, 99, 12000);
    verifier.VerifyFile(obfuscated);

    Check(verifier.GetRenamedFiles(&renamed), "GetRenamedFiles after scanning it again");
    Check(renamed.empty(), "it is no longer reported as renamed");

    std::remove(obfuscated);
    std::remove("restaledir/stale.par2");
    std::remove("restaledir/stale.vol0+4.par2");
  }

  // With no basepath the set is resolved beside its PAR2 files, as the tool
  // does, rather than against the working directory
  {
    Check(MakeDirectory("besidedir"), "mkdir for the derived-basepath check");

    const char *const data = "besidedir/beside.data";
    WriteData(data, 41, 8000);

    std::vector<std::string> files;
    files.emplace_back(data);
    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "besidedir/", 0, 2,
                                             "besidedir/beside", files, BLOCKSIZE, 0,
                                             par2::scUniform, 1, 4),
          "par2create for the derived-basepath check");

    // No basepath, and the PAR2 file named by a path that is not the working
    // directory: without deriving one, every file would look missing.
    par2::Par2Verifier derived(quiet, quiet, par2::nlSilent);
    Check(par2::eSuccess == derived.AddPar2File("besidedir/beside.par2"),
          "AddPar2File with no basepath");

    std::vector<par2::Par2FileInfo> info;
    Check(derived.GetFileInfo(&info), "GetFileInfo with a derived basepath");
    for (const auto &file : info)
    {
      std::ifstream opened(file.localfilename.c_str(), std::ios::binary);
      Check(opened.good(), "localfilename resolves beside the PAR2 file");
    }

    Check(par2::eSuccess == derived.Verify(noextras),
          "the set beside its PAR2 files verifies");

    std::remove(data);
  }

  // A basepath without a trailing separator names the same directory
  {
    // Earlier blocks leave the data files corrupted or missing, so put them
    // back first - WriteData is deterministic, so the set still matches.
    for (size_t i = 0; i < DATACOUNT; ++i)
      WriteData(DATA[i], (unsigned)i, 20000 + i * 9000);

    Check(MakeDirectory("basepathdir"), "mkdir for the basepath check");

    std::vector<std::string> moved;
    for (const char *name : DATA)
    {
      const std::string to = std::string("basepathdir/") + name;
      std::rename(name, to.c_str());
      moved.push_back(to);
    }

    par2::Par2Verifier bare(quiet, quiet, par2::nlSilent, "basepathdir");
    Check(par2::eSuccess == bare.AddPar2File(PARFILE), "AddPar2File for the basepath check");
    Check(par2::eSuccess == bare.Verify(noextras),
          "a basepath with no trailing separator still finds the files");

    // localfilename follows the basepath, separator and all, and is right
    // before anything has been verified because the basepath was known at
    // construction
    std::vector<par2::Par2FileInfo> early;
    par2::Par2Verifier beforeverify(quiet, quiet, par2::nlSilent, "basepathdir");
    Check(par2::eSuccess == beforeverify.AddPar2File(PARFILE), "AddPar2File before verifying");
    Check(beforeverify.GetFileInfo(&early), "GetFileInfo before verifying");
    for (const auto &file : early)
    {
      // Absolute, because the basepath is canonicalised, and ending in the
      // name the set records
      const std::string &local = file.localfilename;
      Check(local.size() > file.filename.size() &&
            local.compare(local.size() - file.filename.size(),
                          std::string::npos, file.filename) == 0,
            "localfilename is right before any verify");
      Check(local.find("basepathdir") != std::string::npos,
            "and sits under the basepath");
    }

    std::vector<par2::Par2FileInfo> located;
    Check(bare.GetFileInfo(&located), "GetFileInfo after a basepath verify");
    Check(located.size() == DATACOUNT, "GetFileInfo count after a basepath verify");
    for (size_t i = 0; i < located.size(); ++i)
    {
      Check(located[i].localfilename == early[i].localfilename,
            "the same path whether or not anything has been verified");

      std::ifstream opened(located[i].localfilename.c_str(), std::ios::binary);
      Check(opened.good(), "and it opens the file it names");
    }

    par2::Par2Verifier slash(quiet, quiet, par2::nlSilent, "basepathdir/");
    Check(par2::eSuccess == slash.AddPar2File(PARFILE), "AddPar2File for the basepath check");
    Check(par2::eSuccess == slash.Verify(noextras),
          "and so does one with it");

    for (size_t i = 0; i < moved.size(); ++i)
      std::rename(moved[i].c_str(), DATA[i]);
  }

  // Naming a set rather than a file, and what eFileIOError actually means
  {
    par2::Par2Verifier byset(quiet, quiet, par2::nlSilent);
    // "consumer" rather than "consumer.par2": the volume files are found
    // beside it and carry the critical packets.
    Check(par2::eSuccess == byset.AddPar2File("consumer"), "a set name is enough");
    par2::Par2SetInfo setinfo;
    Check(byset.GetSetInfo(&setinfo), "naming the set describes it");

    par2::Par2Verifier nothing(quiet, quiet, par2::nlSilent);
    Check(par2::eFileIOError == nothing.AddPar2File("no-such-set-at-all.par2"),
          "a name that yields nothing is eFileIOError");
    CheckLastError(nothing, par2::eFileIOError, "a name that yields nothing");

    par2::Par2Error error;
    Check(nothing.GetLastError(&error), "and it says why");
    Check(error.code == par2::ecPar2FileMissing, "the reason is ecPar2FileMissing");
    Check(error.filename == "no-such-set-at-all.par2", "naming the file it looked for");
  }

  // A set may describe files in subdirectories, so neither name is
  // necessarily a single path component
  {
    Check(MakeDirectory("nestdir"), "mkdir for the subdirectory check");
    Check(MakeDirectory("nestdir/inner"), "mkdir for the nested file");

    const char *const flat = "nestdir/flat.data";
    const char *const nested = "nestdir/inner/nested.data";

    WriteData(flat, 31, 8000);
    WriteData(nested, 32, 8000);

    std::vector<std::string> files;
    files.emplace_back(flat);
    files.emplace_back(nested);

    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "nestdir/", 0, 2,
                                             "nestdir/nest", files, BLOCKSIZE, 0,
                                             par2::scUniform, 1, 4),
          "par2create for the subdirectory check");

    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "nestdir/");
    Check(par2::eSuccess == verifier.AddPar2File("nestdir/nest.par2"),
          "AddPar2File for the subdirectory check");
    Check(par2::eSuccess == verifier.Verify(noextras),
          "a set spanning subdirectories verifies");

    std::vector<par2::Par2FileInfo> nestedinfo;
    Check(verifier.GetFileInfo(&nestedinfo), "GetFileInfo for the subdirectory check");

    bool sawseparator = false;
    for (const auto &file : nestedinfo)
    {
      if (file.filename.find('/') != std::string::npos)
        sawseparator = true;

      std::ifstream opened(file.localfilename.c_str(), std::ios::binary);
      Check(opened.good(), "localfilename opens whatever depth the file is at");
    }
    Check(sawseparator, "a name may hold a directory separator");

    std::remove(flat);
    std::remove(nested);
  }

  // Repair answers rather than crashing when it cannot do the job
  {
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent);
    Check(par2::eLogicError == verifier.Repair(), "Repair needs a verify first");
    CheckLastError(verifier, par2::eLogicError, "Repair without a verify");

    par2::Par2Error error;
    Check(verifier.GetLastError(&error), "Repair without a verify says why");
    Check(error.code == par2::ecNotVerified, "and the reason is ecNotVerified");

    Check(par2::eSuccess == verifier.AddPar2File(PARFILE), "AddPar2File for the repair guard");
    CheckLastError(verifier, par2::eSuccess, "AddPar2File for the repair guard");
    Check(par2::eLogicError == verifier.Repair(),
          "adding packets is still not a verify");
    Check(verifier.GetLastError(&error), "adding packets does not make it verified");
    Check(error.code == par2::ecNotVerified, "so the reason is still ecNotVerified");

    // Lose more blocks than the set can rebuild, so repair is genuinely
    // impossible, then ask for one anyway.
    par2::Par2SetInfo info;
    Check(verifier.GetSetInfo(&info), "GetSetInfo for the repair guard");

    std::vector<par2::Par2FileInfo> files;
    Check(verifier.GetFileInfo(&files), "GetFileInfo for the repair guard");

    for (const auto &file : files)
      std::remove(file.filename.c_str());

    Check(par2::eRepairNotPossible == verifier.Verify(noextras),
          "every file gone is beyond repair");
    CheckLastError(verifier, par2::eRepairNotPossible, "a verify beyond repair");
    Check(par2::eRepairNotPossible == verifier.Repair(),
          "Repair says so too, rather than crashing");
    CheckLastError(verifier, par2::eRepairNotPossible, "a repair that cannot be done");

    // Put them back byte for byte, so the set still matches for anything added
    // after this. WriteData is deterministic, so no new par2create is needed.
    for (size_t i = 0; i < DATACOUNT; ++i)
      WriteData(DATA[i], (unsigned)i, 20000 + i * 9000);
  }

  // Files fed in one at a time as they arrive, including one which is the right
  // size but has not finished downloading when it is first scanned
  {
    Check(MakeDirectory("arrivedir"), "mkdir for the incremental check");

    const char *const arriving[] = {"arrivedir/arrive-0.data",
                                    "arrivedir/arrive-1.data",
                                    "arrivedir/arrive-2.data"};
    const size_t arrivingcount = sizeof(arriving) / sizeof(arriving[0]);

    std::vector<std::string> files;
    for (size_t i = 0; i < arrivingcount; ++i)
    {
      WriteData(arriving[i], (unsigned)(91 + i), 30000);
      files.emplace_back(arriving[i]);
    }
    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "arrivedir/", 0, 2,
                                             "arrivedir/arrive", files, BLOCKSIZE, 0,
                                             par2::scUniform, 1, 20),
          "par2create for the incremental check");

    // Only the third file is on disk, and it is the right size with a hole in
    // the middle, which is what a download in progress looks like
    Corrupt(arriving[2], 12000, 6000);
    std::remove(arriving[0]);
    std::remove(arriving[1]);

    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "arrivedir/");
    Check(par2::eSuccess == verifier.AddPar2File("arrivedir/arrive.par2"),
          "AddPar2File for the incremental check");

    par2::Par2VerifyResult r{};

    // The incomplete one, scanned while it is still a hole
    verifier.VerifyFile(arriving[2]);
    Check(verifier.GetVerifyResult(&r), "GetVerifyResult after the first scan");
    Check(r.completefilecount == 0, "nothing is complete yet");
    Check(r.damagedfilecount == 1, "the file on disk is damaged");
    Check(r.availableblockcount > 0, "but some of its blocks are usable");

    const par2::u32 partial = r.availableblockcount;

    // It finishes downloading, and is scanned again
    WriteData(arriving[2], (unsigned)93, 30000);
    verifier.VerifyFile(arriving[2]);
    Check(verifier.GetVerifyResult(&r), "GetVerifyResult after the rescan");
    Check(r.completefilecount == 1, "the finished file is now complete");
    Check(r.damagedfilecount == 0, "and no longer counts as damaged");
    Check(r.availableblockcount > partial, "the blocks it was missing are there");

    // The other two arrive
    WriteData(arriving[0], (unsigned)91, 30000);
    verifier.VerifyFile(arriving[0]);
    WriteData(arriving[1], (unsigned)92, 30000);
    Check(par2::eSuccess == verifier.VerifyFile(arriving[1]),
          "the set is complete once the last file is scanned");

    Check(verifier.GetVerifyResult(&r), "GetVerifyResult at the end");
    Check(r.completefilecount == arrivingcount, "every file is complete");
    Check(r.missingblockcount == 0, "and nothing is missing");

    for (const char *name : arriving)
      std::remove(name);
  }

  // A PAR2 file fed in along with the data is left alone, since a repair
  // reads the recovery data through the packets read from it
  {
    Check(MakeDirectory("feeddir"), "mkdir for the fed PAR2 file check");

    const char *const data = "feeddir/feed.data";

    WriteData(data, 27, 40000);

    std::vector<std::string> files;
    files.emplace_back(data);
    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "feeddir/", 0, 2,
                                             "feeddir/feed", files, BLOCKSIZE, 0,
                                             par2::scUniform, 1, 8),
          "par2create for the fed PAR2 file check");

    // Named in full, as an application working in absolute paths names them,
    // which is also how the volumes found beside the set are recorded
    std::string dir;
    {
      par2::Par2Verifier probe(quiet, quiet, par2::nlSilent, "feeddir/");
      std::vector<par2::Par2FileInfo> info;
      Check(par2::eSuccess == probe.AddPar2File("feeddir/feed.par2") &&
            probe.GetFileInfo(&info) && info.size() == 1,
            "GetFileInfo for the fed PAR2 file check");
      if (info.size() == 1)
        dir = info[0].localfilename.substr(0, info[0].localfilename.size() -
                                              std::string("feed.data").size());
    }

    Corrupt(data, 1000, 5000);

    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, dir);
    Check(par2::eSuccess == verifier.AddPar2File(dir + "feed.par2"),
          "AddPar2File for the fed PAR2 file check");
    Check(par2::eRepairPossible == verifier.VerifyFile(dir + "feed.data"),
          "the damaged data is found");
    Check(par2::eRepairPossible == verifier.VerifyFile(dir + "feed.vol0+8.par2"),
          "a volume fed in too is not taken for data");
    Check(par2::eRepairPossible == verifier.VerifyFile(dir + "feed.par2"),
          "nor is the index file");
    Check(par2::eSuccess == verifier.Repair(),
          "and the repair still reads its recovery data through them");

    // With the set whole, a file which adds nothing says so as a verify would
    Check(par2::eSuccess == verifier.VerifyFile(dir + "feed.par2"),
          "a PAR2 file fed in once the set is whole reports it whole");
    Check(par2::eSuccess == verifier.VerifyFile(dir + "absent.data"),
          "and so does a file which is not there");

    par2::Par2Verifier after(quiet, quiet, par2::nlSilent, "feeddir/");
    Check(par2::eSuccess == after.AddPar2File("feeddir/feed.par2"),
          "AddPar2File after the fed repair");
    Check(par2::eSuccess == after.Verify(noextras), "the data is whole again");

    std::remove(data);
    std::remove("feeddir/feed.data.1");
    std::remove("feeddir/feed.par2");
    std::remove("feeddir/feed.vol0+8.par2");
  }

  // A data file which arrives before the PAR2 file describing it
  {
    Check(MakeDirectory("firstdir"), "mkdir for the ordering check");

    const char *const early = "firstdir/early.data";
    const char *const late = "firstdir/late.data";

    WriteData(early, 95, 30000);
    WriteData(late, 96, 30000);

    std::vector<std::string> files;
    files.emplace_back(early);
    files.emplace_back(late);
    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "firstdir/", 0, 2,
                                             "firstdir/first", files, BLOCKSIZE, 0,
                                             par2::scUniform, 1, 20),
          "par2create for the ordering check");

    std::remove(late);

    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "firstdir/");

    // Nothing describes it yet, so it cannot be scanned
    Check(par2::eInsufficientCriticalData == verifier.VerifyFile(early),
          "a scan before the set is known says so");
    CheckLastError(verifier, par2::eInsufficientCriticalData, "a scan before the set is known");

    par2::Par2Error unknown;
    Check(verifier.GetLastError(&unknown), "and it says why");
    Check(unknown.code == par2::ecMainPacketMissing, "the reason is ecMainPacketMissing");

    // The PAR2 file arrives and the earlier scan is replayed against it
    Check(par2::eSuccess == verifier.AddPar2File("firstdir/first.par2"),
          "AddPar2File for the ordering check");

    par2::Par2VerifyResult r{};
    Check(verifier.GetVerifyResult(&r), "the replayed scan counts as a verify");
    Check(r.completefilecount == 1, "the file scanned first was found");
    Check(r.missingfilecount == 1, "and the one never scanned is missing");

    std::remove(early);
  }

  // A set which names one file on disk twice says so, rather than racing two
  // threads to claim it
  {
    Check(MakeDirectory("twicedir"), "mkdir for the duplicate check");

    const char *const once = "twicedir/once.data";
    const char *const twicepar = "twicedir/twice.par2";

    WriteData(once, 8, 30000);

    // Named twice, so the set describes the same file on disk under two of
    // its entries
    std::vector<std::string> files;
    files.emplace_back(once);
    files.emplace_back(once);

    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "twicedir/", 0, 2,
                                             "twicedir/twice", files, BLOCKSIZE, 0,
                                             par2::scVariable, 0, RECOVERYBLOCKS),
          "par2create for the duplicate check");

    Counting observer;
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "twicedir/");
    verifier.SetObserver(&observer);

    Check(par2::eSuccess == verifier.AddPar2File(twicepar), "AddPar2File for the duplicate check");

    par2::Par2SetInfo info;
    Check(verifier.GetSetInfo(&info), "GetSetInfo for the duplicate check");
    Check(info.recoverablefilecount == 2, "the set describes the file twice");

    const par2::Result result = verifier.Verify(noextras);
    Check(par2::eFileIOError == result, "a set which names one file twice says so");
    CheckLastError(verifier, result, "a set which names one file twice");

    par2::Par2Error error;
    Check(verifier.GetLastError(&error), "and it says why");
    Check(error.code == par2::ecDuplicateSourceFile, "the reason is ecDuplicateSourceFile");
    Check(error.filename == "once.data", "naming the file both entries point at");
    Check(observer.errors == 1, "reported once rather than for both entries");

    std::remove(once);
    std::remove(twicepar);
  }

  // What went wrong with one file is reported as the operation on that file,
  // rather than as an I/O error with nothing attached
#ifndef _WIN32
  {
    Check(MakeDirectory("rodir"), "mkdir for the read-only check");

    const char *const locked = "rodir/locked.data";
    const char *const lockedpar = "rodir/locked.par2";

    WriteData(locked, 3, 30000);
    std::vector<std::string> files(1, std::string(locked));

    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "rodir/", 0, 2,
                                             lockedpar, files, BLOCKSIZE, 0,
                                             par2::scVariable, 0, 20),
          "par2create for the read-only check");

    Corrupt(locked, 5000, 2000);

    Counting observer;
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "rodir/");
    verifier.SetObserver(&observer);

    Check(par2::eSuccess == verifier.AddPar2File(lockedpar),
          "AddPar2File for the read-only check");
    Check(par2::eRepairPossible == verifier.Verify(noextras),
          "the damaged file needs repairing");

    const bool closed = (chmod("rodir", 0555) == 0) && !CanWriteInto("rodir");

    if (closed)
    {
      const par2::Result result = verifier.Repair();

      Check(par2::eFileIOError == result, "a repair which cannot write says so");
      CheckLastError(verifier, result, "a repair which cannot write");

      par2::Par2Error error;
      Check(verifier.GetLastError(&error), "and it says why");
      Check(error.code == par2::ecFileRenameFailed, "naming the operation on the file");
      Check(error.filename.find("locked.data") != std::string::npos,
            "and the file it was working on");
      Check(observer.errors > 0, "the observer heard about it as it happened");
    }

    chmod("rodir", 0755);

    std::remove(locked);
    std::remove(lockedpar);
  }
#endif

  // The implementations an application supplies reach the work the handle does,
  // rather than being dropped in favour of the ones built in
  {
    Check(MakeDirectory("backends"), "mkdir for the backend check");

    const char *const own = "backends/own.data";
    const char *const ownpar = "backends/own.par2";

    WriteData(own, 9, 30000);
    std::vector<std::string> files(1, std::string(own));

    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "backends/", 0, 2,
                                             ownpar, files, BLOCKSIZE, 0,
                                             par2::scVariable, 0, 20),
          "par2create for the backend check");

    Corrupt(own, 5000, 2000);

    int asked = 0;
    par2::u32 budget = 0;
    par2::Backends backends;
    backends.processor = [&asked, &budget](const par2::ProcessorConfig &config)
    {
      ++asked;
      budget = config.numthreads;
      return std::unique_ptr<par2::Processor>();
    };

    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "backends/", backends);

    // What the handle was told, rather than whatever the library would pick
    verifier.SetThreadCounts(3, 1);

    Check(par2::eSuccess == verifier.AddPar2File(ownpar), "AddPar2File for the backend check");
    Check(par2::eRepairPossible == verifier.Verify(noextras),
          "the damaged file needs repairing");

    // Declining to supply one is reported rather than quietly falling back
    Check(par2::eMemoryError == verifier.Repair(),
          "a repair which cannot build a processor says so");
    Check(asked == 1, "the application's processor was asked for");
    Check(budget == 3, "and built with the thread count the handle was given");
    CheckLastError(verifier, par2::eMemoryError, "a repair with no processor");

    // The application's own code failing is not the disk failing
    par2::Par2Error noprocessor;
    Check(verifier.GetLastError(&noprocessor), "and it says why");
    Check(noprocessor.code == par2::ecProcessorFailed, "the reason is ecProcessorFailed");

    std::remove(own);
    std::remove(ownpar);
  }

  // A set built through the handle is the same set, and the observer hears
  // about the work even at nlSilent
  {
    Check(MakeDirectory("madedir"), "mkdir for the create handle check");

    const char *const made = "madedir/made.data";
    const char *const madepar = "madedir/made.par2";

    WriteData(made, 5, 40000);

    Counting observer;
    par2::Par2Creator creator(quiet, quiet, par2::nlSilent, "madedir/");
    creator.SetObserver(&observer);
    creator.AddSourceFile(made);
    creator.SetBlockSize(BLOCKSIZE);
    creator.SetRecoveryBlockCount(RECOVERYBLOCKS);
    creator.SetRecoveryFileScheme(par2::scVariable);

    // Zero for either of these leaves it at the library's own default
    creator.SetMemoryLimit(0);
    creator.SetThreadCounts(0, 0);

    Check(par2::eSuccess == creator.Create(madepar), "Create through the handle");
    CheckLastError(creator, par2::eSuccess, "a create which worked");

    par2::Par2Error error;
    Check(!creator.GetLastError(&error), "a create which worked reports no error");

    Check(observer.files == 1, "OnFile for the source file");
    Check(observer.done == 1, "and an OnFileDone to pair with it");
    Check(observer.setinfo == 1, "OnSetInfo once the set is known");
    Check(observer.lastinfo.recoverablefilecount == 1, "OnSetInfo file count");
    Check(observer.lastinfo.blocksize == BLOCKSIZE, "OnSetInfo blocksize");
    Check(observer.lastinfo.recoveryblocks == RECOVERYBLOCKS, "OnSetInfo recovery blocks");
    Check(observer.lastinfo.datasize == 40000, "OnSetInfo data size");

    // The guards which used to stop this reaching the meter are gone
    Check(observer.progress > 0, "OnProgress even at nlSilent");
    Check(observer.reached, "and it reaches a thousand");

    // The name it was given is the name of the set, not of a set called after it
    {
      std::ifstream twice("madedir/made.par2.par2", std::ios::binary);
      Check(!twice.is_open(), "the .par2 suffix was not applied twice");
    }

    // The set it wrote is a set
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "madedir/");
    Check(par2::eSuccess == verifier.AddPar2File(madepar), "AddPar2File for the made set");
    Check(par2::eSuccess == verifier.Verify(noextras), "the made set verifies clean");

    std::remove(made);
    std::remove(madepar);
  }

  // A create asked to stop leaves nothing of the set behind, whether it is
  // stopped while it reads the source files or while it writes the set
  {
    Check(MakeDirectory("stopdir"), "mkdir for the create cancel check");

    const char *const stopped = "stopdir/stopped.data";
    const char *const stoppedpar = "stopdir/stopped.par2";

    WriteData(stopped, 6, 400000);

    // The phase says which pass a progress report belongs to, so the cancel can
    // be aimed at the one which reads the source files or at the one which
    // writes the recovery data
    class Stopper : public par2::Par2Observer
    {
    public:
      par2::Par2Creator *creator;
      par2::Phase wanted;
      bool landed;
      explicit Stopper(par2::Phase wanted) : creator(0), wanted(wanted), landed(false) {}
      void OnProgress(par2::Phase phase, par2::u32) override
      {
        if (creator && phase == wanted)
        {
          landed = true;
          creator->Cancel();
        }
      }
    };

    for (int late = 0; late < 2; ++late)
    {
      const std::string when = late ? "while writing" : "while reading";

      Stopper stopper(late ? par2::phProcessing : par2::phHashing);
      par2::Par2Creator creator(quiet, quiet, par2::nlSilent, "stopdir/");
      stopper.creator = &creator;
      creator.SetObserver(&stopper);
      creator.AddSourceFile(stopped);
      creator.SetBlockSize(BLOCKSIZE);
      creator.SetRecoveryBlockCount(RECOVERYBLOCKS);
      creator.SetRecoveryFileScheme(par2::scVariable);

      // Too little for one pass, so the source files are hashed as they are
      // read and there is progress to cancel on during that
      creator.SetMemoryLimit(BLOCKSIZE * 4);

      Check(par2::eCancelled == creator.Create(stoppedpar), "Create is cancelled " + when);
      CheckLastError(creator, par2::eCancelled, "a cancelled create " + when);
      Check(stopper.landed, "the cancel landed in the phase it was meant to");

      par2::Par2Verifier gone(quiet, quiet, par2::nlSilent, "stopdir/");
      Check(par2::eFileIOError == gone.AddPar2File(stoppedpar),
            "and no PAR2 file was left behind " + when);
    }

    // The same handle creates a whole set once the request is withdrawn
    par2::Par2Creator creator(quiet, quiet, par2::nlSilent, "stopdir/");
    creator.AddSourceFile(stopped);
    creator.SetBlockSize(BLOCKSIZE);
    creator.SetRecoveryBlockCount(RECOVERYBLOCKS);
    creator.SetRecoveryFileScheme(par2::scVariable);
    creator.Cancel();
    creator.ClearCancel();

    Check(par2::eSuccess == creator.Create(stoppedpar), "Create again after clearing");

    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "stopdir/");
    Check(par2::eSuccess == verifier.AddPar2File(stoppedpar), "AddPar2File for the second set");
    Check(par2::eSuccess == verifier.Verify(noextras), "the second set verifies clean");

    std::remove(stopped);
    std::remove(stoppedpar);
  }

  // A create says what was wrong with what it was asked to do
  {
    Check(MakeDirectory("baddir"), "mkdir for the create error check");

    const char *const bad = "baddir/bad.data";
    const char *const badpar = "baddir/bad.par2";

    WriteData(bad, 7, 30000);

    // A block size which is not a multiple of four
    {
      Counting observer;
      par2::Par2Creator creator(quiet, quiet, par2::nlSilent, "baddir/");
      creator.SetObserver(&observer);
      creator.AddSourceFile(bad);
      creator.SetBlockSize(BLOCKSIZE + 1);
      creator.SetRecoveryBlockCount(RECOVERYBLOCKS);

      const par2::Result result = creator.Create(badpar);
      Check(par2::eInvalidCommandLineArguments == result, "an unusable block size says so");
      CheckLastError(creator, result, "an unusable block size");

      par2::Par2Error error;
      Check(creator.GetLastError(&error), "and it says why");
      Check(error.code == par2::ecInvalidSetting, "the reason is ecInvalidSetting");
      Check(observer.errors == 1, "the observer heard about it too");
    }

    // Declining to supply a processor is reported rather than quietly falling back
    {
      int asked = 0;
      par2::u32 budget = 0;
      par2::Backends backends;
      backends.processor = [&asked, &budget](const par2::ProcessorConfig &config)
      {
        ++asked;
        budget = config.numthreads;
        return std::unique_ptr<par2::Processor>();
      };

      par2::Par2Creator creator(quiet, quiet, par2::nlSilent, "baddir/", backends);
      creator.AddSourceFile(bad);
      creator.SetBlockSize(BLOCKSIZE);
      creator.SetRecoveryBlockCount(RECOVERYBLOCKS);
      creator.SetThreadCounts(3, 1);

      const par2::Result result = creator.Create(badpar);
      Check(par2::eMemoryError == result, "a create which cannot build a processor says so");
      Check(asked == 1, "the application's processor was asked for");
      Check(budget == 3, "and built with the thread count the handle was given");
      CheckLastError(creator, result, "a create with no processor");

      par2::Par2Error error;
      Check(creator.GetLastError(&error), "and it says why");
      Check(error.code == par2::ecProcessorFailed, "the reason is ecProcessorFailed");

      // What it had already written is taken away again
      par2::Par2Verifier gone(quiet, quiet, par2::nlSilent, "baddir/");
      Check(par2::eFileIOError == gone.AddPar2File(badpar),
            "and a create which failed leaves no set behind");
    }

    // A source file outside the basepath cannot be named relative to it
    {
      Counting observer;
      par2::Par2Creator creator(quiet, quiet, par2::nlSilent, "baddir/");
      creator.SetObserver(&observer);
      creator.AddSourceFile(DATA[1]);
      creator.SetBlockSize(BLOCKSIZE);
      creator.SetRecoveryBlockCount(RECOVERYBLOCKS);

      const par2::Result result = creator.Create(badpar);
      Check(par2::eInvalidCommandLineArguments == result, "a file outside the basepath says so");
      CheckLastError(creator, result, "a file outside the basepath");

      par2::Par2Error error;
      Check(creator.GetLastError(&error), "and it says why");
      Check(error.code == par2::ecInvalidSetting, "the reason is ecInvalidSetting");
      Check(error.filename.find(DATA[1]) != std::string::npos, "naming the file");
      Check(observer.errors == 1, "the observer heard about it too");

      par2::Par2Verifier none(quiet, quiet, par2::nlSilent, "baddir/");
      Check(par2::eFileIOError == none.AddPar2File(badpar), "and nothing was written");
    }

    // With no basepath given, each Create takes the directory of its own set
    {
      Check(MakeDirectory("baddir/sub"), "mkdir for the per-set basepath check");

      const char *const inner = "baddir/sub/inner.data";
      WriteData(inner, 9, 20000);

      par2::Par2Creator creator(quiet, quiet, par2::nlSilent);
      creator.AddSourceFile(inner);
      creator.SetBlockSize(BLOCKSIZE);
      creator.SetRecoveryBlockCount(4);
      creator.SetRecoveryFileScheme(par2::scUniform, 1);

      Check(par2::eSuccess == creator.Create("baddir/outer.par2"),
            "the first set, above the file");
      Check(par2::eSuccess == creator.Create("baddir/sub/inner.par2"),
            "and the second, beside it");

      par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent);
      std::vector<par2::Par2FileInfo> info;
      Check(par2::eSuccess == verifier.AddPar2File("baddir/sub/inner.par2") &&
            verifier.GetFileInfo(&info),
            "GetFileInfo for the second set");
      Check(info.size() == 1 && info[0].filename == "inner.data",
            "whose name is relative to its own directory, not the first set's");

      std::remove(inner);
      std::remove("baddir/outer.par2");
      std::remove("baddir/outer.vol0+4.par2");
      std::remove("baddir/sub/inner.par2");
      std::remove("baddir/sub/inner.vol0+4.par2");
    }

    std::remove(bad);
  }

  // A handle built without streams, which is how an application that shows the
  // work itself uses the library
  {
    Check(MakeDirectory("streamlessdir"), "mkdir for the streamless check");

    const char *const source = "streamlessdir/streamless.data";
    const char *const set = "streamlessdir/streamless.par2";

    WriteData(source, 37, 60000);

    // Whatever the library would have written, nothing reaches the stream the
    // rest of this test shares
    const size_t written = quiet.str().size();

    Counting creating;
    {
      par2::Par2Creator creator("streamlessdir/");
      creator.SetObserver(&creating);
      creator.AddSourceFile(source);
      creator.SetBlockSize(BLOCKSIZE);
      creator.SetRecoveryBlockCount(RECOVERYBLOCKS);
      creator.SetMemoryLimit(BLOCKSIZE * 4);

      Check(par2::eSuccess == creator.Create(set), "a create with no streams");
    }

    Check(creating.files == 1, "the observer saw the file being hashed");
    Check(creating.done == 1, "and saw it finish");
    Check(creating.progress > 0, "and was told how far along it was");
    Check(creating.reached, "and saw the work reach the end");
    Check(creating.setinfo == 1, "and was told what the set is");

    // A create hashes, builds a matrix and computes, in that order. It never
    // solves one, having no missing blocks to solve for.
    Check(creating.phases.size() == 3, "a create reports three steps");
    Check(creating.phases[0] == par2::phHashing, "the source files first");
    Check(creating.phases[1] == par2::phConstructing, "then the matrix");
    Check(creating.phases[2] == par2::phProcessing, "then the recovery data");
    Check(!creating.Saw(par2::phSolving), "and nothing to solve");

    Corrupt(source, 20000, 5000);

    Counting repairing;
    int scanned = 0;
    {
      par2::Par2Verifier verifier("streamlessdir/");
      verifier.SetObserver(&repairing);

      Check(par2::eSuccess == verifier.AddPar2File(set), "AddPar2File with no streams");

      // The PAR2 files are announced too, so only what arrives after them
      // belongs to the set's own file
      const int par2files = repairing.files;

      Check(par2::eRepairPossible == verifier.Verify(noextras),
            "a verify with no streams still finds the damage");

      scanned = repairing.files - par2files;

      Check(par2::eSuccess == verifier.Repair(), "and the repair works");
    }

    Check(scanned == 1, "the observer saw the file being scanned");
    Check(repairing.files == repairing.done, "and every file reported is a file finished");
    Check(repairing.progress > 0, "and was told how far along it was");

    // A repair which rebuilds from recovery data works a matrix out first, and
    // reads back what it wrote unless it was told not to
    Check(repairing.Saw(par2::phConstructing), "the matrix was built");
    Check(repairing.Saw(par2::phSolving), "and solved, because blocks were missing");
    Check(repairing.Saw(par2::phProcessing), "and the missing blocks rebuilt");
    Check(repairing.Saw(par2::phVerifyingRepair), "and the result read back");

    // This one is written to serr by a handle which has one, whatever its
    // NoiseLevel. Without streams it is only recorded.
    {
      const char *const notaset = "streamlessdir/notaset.par2";
      WriteData(notaset, 11, 5000);

      par2::Par2Verifier verifier("streamlessdir/");
      Check(par2::eInsufficientCriticalData == verifier.AddPar2File(notaset),
            "AddPar2File on a file which is not a set");

      par2::Par2Error error;
      Check(verifier.GetLastError(&error), "and it says why without a stream to say it on");
      Check(error.code == par2::ecMainPacketMissing, "the reason is ecMainPacketMissing");

      std::remove(notaset);
    }

    Check(quiet.str().size() == written, "and nothing at all was written to a stream");

    std::remove(source);
    std::remove(set);
  }

  // A name the set has to record is reported when this system may not take it
  // back, without the work stopping
#ifndef _WIN32
  {
    Check(MakeDirectory("warndir"), "mkdir for the warning check");

    const char *const odd = "warndir/what?now.data";
    const char *const oddpar = "warndir/what.par2";

    WriteData(odd, 13, 30000);

    Counting observer;
    par2::Par2Creator creator("warndir/");
    creator.SetObserver(&observer);
    creator.AddSourceFile(odd);
    creator.SetBlockSize(BLOCKSIZE);
    creator.SetRecoveryBlockCount(RECOVERYBLOCKS);

    Check(par2::eSuccess == creator.Create(oddpar),
          "an awkward name does not stop a create");
    Check(observer.errors == 0, "and is not an error");
    Check(observer.warnings > 0, "but it is reported");
    Check(observer.lastwarning.code == par2::wcFilenameUnsafe,
          "as a name some systems will not take");
    Check(observer.lastwarning.filename == "what?now.data",
          "naming the file it concerns, as the set records it");
    Check(observer.lastwarning.message.find('?') != std::string::npos,
          "and saying what about the name");

    std::remove(odd);
  }
#endif

  for (const char *name : DATA)
    std::remove(name);

  if (failures != 0)
    return 1;

  std::cout << "SUCCESS: consumer complete." << std::endl;

  return 0;
}
