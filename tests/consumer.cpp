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
#include <fstream>
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

  void WriteData(const char *name, unsigned seed, size_t bytes)
  {
    std::ofstream f(name, std::ios::binary | std::ios::trunc);
    for (size_t i = 0; i < bytes; ++i)
      f.put((char)((i * 31 + seed * 7) & 0xff));
  }

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
      files.push_back(DATA[i]);
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
class Counting : public par2::Par2Observer
{
public:
  Counting()
    : setinfo(0), files(0), progress(0), done(0), repairs(0),
      lastinfo(), last(0), wentbackwards(false), reached(false) {}

  int setinfo, files, progress, done, repairs;

  // What the last OnSetInfo carried
  par2::Par2SetInfo lastinfo;

  // Enough to tell one run of progress from several
  par2::u32 last;
  bool wentbackwards, reached;

  void OnSetInfo(const par2::Par2SetInfo &info)
  {
    std::lock_guard<std::mutex> lock(mutex);
    ++setinfo;
    lastinfo = info;
  }

  void OnFile(const std::string &)
  {
    std::lock_guard<std::mutex> lock(mutex);
    ++files;
  }

  void OnFileDone(const std::string &, par2::u32, par2::u32)
  {
    std::lock_guard<std::mutex> lock(mutex);
    ++done;
  }

  void OnRepairStart(void)
  {
    std::lock_guard<std::mutex> lock(mutex);
    ++repairs;
  }

  void OnProgress(par2::u32 permille)
  {
    std::lock_guard<std::mutex> lock(mutex);

    ++progress;
    if (permille < last)
      wentbackwards = true;
    if (permille == 1000)
      reached = true;
    last = permille;
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

    par2::Par2SetInfo info;
    Check(verifier.GetSetInfo(&info), "GetSetInfo");
    Check(info.blocksize == BLOCKSIZE, "GetSetInfo blocksize");
    Check(info.recoverablefilecount == DATACOUNT, "GetSetInfo file count");
    bool setidset = false;
    for (size_t i = 0; i < info.setid.size(); ++i)
      setidset = setidset || info.setid[i] != 0;
    Check(setidset, "GetSetInfo setid");
    // Counted from the packets, so it reads before anything has been verified
    Check(info.recoveryblocks == RECOVERYBLOCKS, "GetSetInfo recovery blocks");

    Check(observer.setinfo == 1, "OnSetInfo once");
    Check(observer.lastinfo.recoveryblocks == RECOVERYBLOCKS,
          "OnSetInfo recovery blocks");
    Check(observer.lastinfo.datablocks == info.datablocks, "OnSetInfo data blocks");
    Check(observer.lastinfo.blocksize == info.blocksize, "OnSetInfo blocksize");

    std::vector<par2::Par2FileInfo> files;
    Check(verifier.GetFileInfo(&files), "GetFileInfo");
    Check(files.size() == DATACOUNT, "GetFileInfo count");
    for (size_t i = 0; i < files.size(); ++i)
    {
      Check(files[i].hash16k.size() == 16, "GetFileInfo hash16k");
      Check(files[i].hashfull.size() == 16, "GetFileInfo hashfull");
      Check(files[i].hash16k != files[i].hashfull, "the two hashes differ");
    }

    par2::u32 totalblocks = 0;
    for (size_t i = 0; i < files.size(); ++i)
    {
      Check(files[i].blockcount > 0, "GetFileInfo blockcount");
      totalblocks += files[i].blockcount;
    }
    Check(totalblocks == info.datablocks, "GetFileInfo blocks add up");

    for (size_t i = 0; i < files.size(); ++i)
    {
      std::vector<par2::u32> crcs;
      Check(verifier.GetBlockChecksums(files[i].filename, &crcs),
            "GetBlockChecksums");
      Check(crcs.size() == files[i].blockcount,
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
    for (size_t i = 0; i < files.size(); ++i)
    {
      std::vector<bool> found;
      Check(verifier.GetFoundBlocks(files[i].filename, &found), "GetFoundBlocks");
      Check(found.size() == files[i].blockcount,
            "GetFoundBlocks one entry per block");
      for (size_t b = 0; b < found.size(); ++b)
        foundblocks += found[b] ? 1 : 0;
    }
    {
      std::vector<bool> found;
      Check(!verifier.GetFoundBlocks("not-in-the-set.bin", &found),
            "GetFoundBlocks rejects an unknown name");
      Check(found.empty(), "GetFoundBlocks clears on failure");
    }

    par2::Par2VerifyResult status;
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
      files.push_back(away[i]);
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
    for (size_t i = 0; i < reported.size(); ++i)
    {
      Check(reported[i].filename.find('/') == std::string::npos, "no path in the name");
      Check(reported[i].filename.find('\\') == std::string::npos, "no separator in the name");
    }

    // and known blocks are keyed by those same names
    for (size_t i = 0; i < reported.size(); ++i)
      verifier.SetKnownBlocks(reported[i].filename,
                              std::vector<bool>(reported[i].blockcount, true));

    Check(par2::eSuccess == verifier.Verify(noextras),
          "known blocks are keyed by the reported name");

    for (size_t i = 0; i < 2; ++i)
      std::remove(away[i]);
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
    for (size_t b = 0; b < damaged.size(); ++b)
      missing += damaged[b] ? 0 : 1;
    Check(missing > 0, "the damaged block is not found in the file");
    Check(missing < damaged.size(), "the undamaged ones still are");

    // The set's files are shifts of one periodic sequence, so the damaged
    // block turns up elsewhere and the repair does not have to rebuild it
    par2::Par2VerifyResult damagedstatus;
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
    Check(observer.repairs == 1, "OnRepairStart called");

    std::vector<bool> repaired;
    Check(verifier.GetFoundBlocks(damagedfiles[which].filename, &repaired),
          "GetFoundBlocks after a repair which read back what it wrote");
    size_t stillmissing = 0;
    for (size_t b = 0; b < repaired.size(); ++b)
      stillmissing += repaired[b] ? 0 : 1;
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
    for (size_t i = 0; i < files.size(); ++i)
      trusting.SetKnownBlocks(files[i].filename,
                              std::vector<bool>(files[i].blockcount, true));

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

    par2::Par2VerifyResult writtenoffstatus;
    Check(writtenoff.GetVerifyResult(&writtenoffstatus), "GetVerifyResult after writing a file off");
    Check(writtenoffstatus.missingblockcount >= all[0].blockcount,
          "its blocks are counted as missing");
  }

  // Cancelling stops the work and says so
  {
    class Canceller : public par2::Par2Observer
    {
    public:
      par2::Par2Verifier *verifier;
      Canceller() : verifier(0) {}
      void OnProgress(par2::u32) { if (verifier) verifier->Cancel(); }
    };

    Canceller canceller;
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent);
    canceller.verifier = &verifier;
    verifier.SetObserver(&canceller);

    Check(par2::eSuccess == verifier.AddPar2File(PARFILE), "AddPar2File before cancelling");
    Check(par2::eCancelled == verifier.Verify(noextras), "Verify is cancelled");

    verifier.ClearCancel();
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
      files.push_back(data[i]);
    }

    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "", 0, 2,
                                             "stream", files, BLOCKSIZE, 0,
                                             par2::scUniform, 8, 24),
          "par2create for the streaming set");

    // none of the recovery files have arrived yet
    for (size_t i = 0; i < volcount; ++i)
      std::rename(vols[i], (std::string("held-") + vols[i]).c_str());

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

    for (size_t i = 0; i < 2; ++i)
      std::remove(data[i]);
  }

  // Reassess before anything has been verified says so
  {
    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent);
    Check(par2::eLogicError == verifier.Reassess(), "Reassess needs a verify first");
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
    files.push_back(data);
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

    par2::Par2VerifyResult partial;
    Check(verifier.GetVerifyResult(&partial), "GetVerifyResult for the partial volume");

    // The rest of the volume arrives
    std::ofstream rest(vol, std::ios::binary | std::ios::trunc);
    rest.write(bytes.data(), (std::streamsize)bytes.size());
    rest.close();

    Check(par2::eSuccess == verifier.AddPar2File(vol), "AddPar2File for the completed volume");
    Check(par2::eLogicError != verifier.Reassess(), "Reassess after the volume completed");

    par2::Par2VerifyResult complete;
    Check(verifier.GetVerifyResult(&complete), "GetVerifyResult after completion");
    Check(complete.recoveryblockcount > partial.recoveryblockcount,
          "the packets written since are picked up");

    std::remove(data);
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
      files.push_back(observed[i]);
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
    Check(observer.progress == 0, "AddPar2File reports no progress");

    const int par2files = observer.files;

    Check(par2::eSuccess == verifier.Verify(noextras), "Verify for the observer check");

    Check(observer.progress > 1, "a scan reports progress more than once");
    Check(!observer.wentbackwards, "and it only ever goes up");
    Check(observer.reached, "reaching the end");
    Check(observer.files == observer.done, "every file reported is a file finished");
    Check(observer.files - par2files == (int)observedcount,
          "one report per file in the set, on top of the PAR2 files");

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
    files.push_back(data);
    Check(par2::eSuccess == par2::par2create(quiet, quiet, par2::nlSilent,
                                             64 * 1024 * 1024, "skipdir/", 0, 2,
                                             "skipdir/skip", files, BLOCKSIZE, 0,
                                             par2::scUniform, 1, 8),
          "par2create for the skipped-verification check");

    Corrupt(data, 500, 400);

    par2::Par2Verifier verifier(quiet, quiet, par2::nlSilent, "skipdir/");
    Check(par2::eSuccess == verifier.AddPar2File("skipdir/skip.par2"),
          "AddPar2File for the skipped-verification check");
    Check(par2::eRepairPossible == verifier.Verify(noextras),
          "the damage is repairable");

    Check(par2::eSuccess == verifier.Repair(false), "Repair without reading it back");

    // Nothing recounted the files, so the numbers still describe the damage
    par2::Par2VerifyResult stale;
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

  // A file found under another name is reported as a pair, and stays reported
  // after the repair that renames it into place
  {
    Check(MakeDirectory("renamedir"), "mkdir for the rename check");

    const char *const data = "renamedir/proper.data";
    const char *const obfuscated = "renamedir/9f3ac1b7e2.dat";

    WriteData(data, 51, 12000);

    std::vector<std::string> files;
    files.push_back(data);
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
    extras.push_back(obfuscated);
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

  // With no basepath the set is resolved beside its PAR2 files, as the tool
  // does, rather than against the working directory
  {
    Check(MakeDirectory("besidedir"), "mkdir for the derived-basepath check");

    const char *const data = "besidedir/beside.data";
    WriteData(data, 41, 8000);

    std::vector<std::string> files;
    files.push_back(data);
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
    for (size_t i = 0; i < info.size(); ++i)
    {
      std::ifstream opened(info[i].localfilename.c_str(), std::ios::binary);
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
    for (size_t i = 0; i < DATACOUNT; ++i)
    {
      const std::string to = std::string("basepathdir/") + DATA[i];
      std::rename(DATA[i], to.c_str());
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
    for (size_t i = 0; i < early.size(); ++i)
    {
      // Absolute, because the basepath is canonicalised, and ending in the
      // name the set records
      const std::string &local = early[i].localfilename;
      Check(local.size() > early[i].filename.size() &&
            local.compare(local.size() - early[i].filename.size(),
                          std::string::npos, early[i].filename) == 0,
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
    files.push_back(flat);
    files.push_back(nested);

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
    for (size_t i = 0; i < nestedinfo.size(); ++i)
    {
      if (nestedinfo[i].filename.find('/') != std::string::npos)
        sawseparator = true;

      std::ifstream opened(nestedinfo[i].localfilename.c_str(), std::ios::binary);
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

    Check(par2::eSuccess == verifier.AddPar2File(PARFILE), "AddPar2File for the repair guard");
    Check(par2::eLogicError == verifier.Repair(),
          "adding packets is still not a verify");

    // Lose more blocks than the set can rebuild, so repair is genuinely
    // impossible, then ask for one anyway.
    par2::Par2SetInfo info;
    Check(verifier.GetSetInfo(&info), "GetSetInfo for the repair guard");

    std::vector<par2::Par2FileInfo> files;
    Check(verifier.GetFileInfo(&files), "GetFileInfo for the repair guard");

    for (size_t i = 0; i < files.size(); ++i)
      std::remove(files[i].filename.c_str());

    Check(par2::eRepairNotPossible == verifier.Verify(noextras),
          "every file gone is beyond repair");
    Check(par2::eRepairNotPossible == verifier.Repair(),
          "Repair says so too, rather than crashing");

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
      files.push_back(arriving[i]);
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

    par2::Par2VerifyResult r;

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

    for (size_t i = 0; i < arrivingcount; ++i)
      std::remove(arriving[i]);
  }

  // A data file which arrives before the PAR2 file describing it
  {
    Check(MakeDirectory("firstdir"), "mkdir for the ordering check");

    const char *const early = "firstdir/early.data";
    const char *const late = "firstdir/late.data";

    WriteData(early, 95, 30000);
    WriteData(late, 96, 30000);

    std::vector<std::string> files;
    files.push_back(early);
    files.push_back(late);
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

    // The PAR2 file arrives and the earlier scan is replayed against it
    Check(par2::eSuccess == verifier.AddPar2File("firstdir/first.par2"),
          "AddPar2File for the ordering check");

    par2::Par2VerifyResult r;
    Check(verifier.GetVerifyResult(&r), "the replayed scan counts as a verify");
    Check(r.completefilecount == 1, "the file scanned first was found");
    Check(r.missingfilecount == 1, "and the one never scanned is missing");

    std::remove(early);
  }

  for (size_t i = 0; i < DATACOUNT; ++i)
    std::remove(DATA[i]);

  if (failures != 0)
    return 1;

  std::cout << "SUCCESS: consumer complete." << std::endl;

  return 0;
}
