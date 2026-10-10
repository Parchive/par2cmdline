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

#include <par2/cli.h>

#include "commandline.h"

#ifdef _WIN32
#include "wargs.h"
#endif

#include <algorithm>
#include <new>
#include <set>

namespace par2
{

// Say on serr what the exception being handled stopped, and return it as the
// failure of the run
static Result Thrown(std::ostream &serr)
{
  try
  {
    throw;
  }
  catch (const std::bad_alloc &)
  {
    serr << "Memory ran out." << std::endl;
    return eMemoryError;
  }
  catch (...)
  {
    serr << "The work stopped on an exception." << std::endl;
    return eLogicError;
  }
}

// The last component of a path
static std::string NameOf(const std::string &filename)
{
  std::string path;
  std::string name;
  DiskFile::SplitFilename(filename, path, name);
  return name;
}

// The detail of the work a noise level prints
static Verbosity VerbosityAt(const NoiseLevel noiselevel)
{
  if (noiselevel >= nlDebug)
    return vbDebug;
  if (noiselevel >= nlNoisy)
    return vbVerbose;
  return vbNone;
}

// Prints what the work does, from what the observer is told of it, at the
// noise level the command line asked for. The callbacks arrive from the
// threads doing the work, several at a time.
class Printer : public Par2Observer
{
public:
  Printer(std::ostream &_sout, std::ostream &_serr, const NoiseLevel _noiselevel)
  : sout(_sout)
  , serr(_serr)
  , noiselevel(_noiselevel)
  , mutex()
  , started(false)
  , phase(phLoading)
  , basepath()
  , repairing(false)
  , sourcefilecount(0)
  , scanned(0)
  , keeploaded(false)
  , loaded()
  , createsummary()
  , blocksize(0)
  , reached(0)
  , opening()
  , open(0)
  , sourcedone(0)
  , sourcecomplete(0)
  , extraheader(false)
  , duplicates()
  , writtenbytes(0)
  , writtenblocks(0)
  {
  }

  // Whether what is printed from noise level least is shown
  bool Shows(const NoiseLevel least) const
  {
    return noiselevel >= least;
  }

  // Print text as it stands, from noise level least
  void Out(const NoiseLevel least, const std::string &text)
  {
    std::lock_guard<std::mutex> lock(mutex);
    Write(sout, least, text);
  }

  void Err(const NoiseLevel least, const std::string &text)
  {
    std::lock_guard<std::mutex> lock(mutex);
    Write(serr, least, text);
  }

  // The directory the names of the files scanned are printed relative to
  void SetBasePath(const std::string &_basepath)
  {
    basepath = _basepath;
  }

  // Whether the processing a repair does rebuilds missing blocks
  void SetRepairing(const bool _repairing)
  {
    repairing = _repairing;
  }

  // How many files of the set a verify scans before any extra files
  void SetSourceFileCount(const size_t count)
  {
    sourcefilecount = count;
    scanned = 0;
    sourcedone = 0;
    sourcecomplete = 0;
    extraheader = false;
  }

  // A verify has come to a result: when it went on to the extra files and
  // there were none to scan, say so as it would have with some
  void Verified(const u32 recoverablefilecount)
  {
    std::lock_guard<std::mutex> lock(mutex);

    if (!extraheader && sourcecomplete < recoverablefilecount)
      Write(sout, nlNormal, "\nScanning extra files:\n\n");
  }

  // Whether to keep the names of the PAR2 files read, which are the ones a
  // purge deletes
  void KeepLoaded(const bool keep)
  {
    keeploaded = keep;
  }

  const std::vector<std::string> &Loaded(void) const
  {
    return loaded;
  }

  // What a create prints once it starts reading the files
  void SetCreateSummary(const std::string &summary)
  {
    createsummary = summary;
  }

  // The block size of a create, which is what the recovery blocks it wrote
  // come to
  void SetBlockSize(const u64 _blocksize)
  {
    blocksize = _blocksize;
  }

  // The step in progress is over
  void Finish(void)
  {
    std::lock_guard<std::mutex> lock(mutex);
    Leave();
  }

  void OnFile(Phase _phase, const std::string &filename) override
  {
    std::lock_guard<std::mutex> lock(mutex);
    Enter(_phase);

    switch (_phase)
    {
    case phLoading:
      Write(sout, nlQuiet, "Loading \"" + NameOf(filename) + "\".\n");
      if (keeploaded)
        loaded.push_back(filename);
      break;
    case phHashing:
      Write(sout, nlQuiet, "Opening: " + filename + "\n");
      break;
    case phScanning:
      if (++scanned == sourcefilecount + 1)
      {
        Write(sout, nlNormal, "\nScanning extra files:\n\n");
        extraheader = true;
      }
      opening.push_back(filename);
      ++open;
      break;
    case phVerifyingRepair:
      opening.push_back(filename);
      ++open;
      break;
    default:
      break;
    }
  }

  void OnProgress(Phase _phase, u32 permille) override
  {
    std::lock_guard<std::mutex> lock(mutex);
    Enter(_phase);

    reached = permille;

    // The files started on are being read now
    for (const auto &name : opening)
      Write(sout, nlNormal, "Opening: \"" + Shortened(name) + "\"\n");
    opening.clear();

    if (Shows(nlNormal))
      sout << Label(_phase) << permille/10 << '.' << permille%10 << "%\r" << std::flush;
  }

  void OnFileDone(Phase _phase, const Par2FileResult &result) override
  {
    std::lock_guard<std::mutex> lock(mutex);
    Enter(_phase);

    switch (_phase)
    {
    case phLoading:
      if (result.packetsfound > 0)
      {
        std::string loaded = "Loaded " + std::to_string(result.packetsfound) + " new packets";
        if (result.blocksfound > 0)
          loaded += " including " + std::to_string(result.blocksfound) + " recovery blocks";
        Write(sout, nlNormal, loaded + "\n");
      }
      else
      {
        Write(sout, nlNormal, "No new packets found\n");
      }
      break;
    case phScanning:
    case phVerifyingRepair:
      {
        --open;

        // A file only opened if there was something in it to read
        auto started = std::find(opening.begin(), opening.end(), result.filename);
        if (started != opening.end())
        {
          opening.erase(started);
          if (result.exists && result.filesize > 0 && 0 == duplicates.count(result.localfilename)
              && !(result.complete && !result.scanned))
            Write(sout, nlNormal, "Opening: \"" + Shortened(result.filename) + "\"\n");
        }

        if (_phase == phScanning && sourcedone < sourcefilecount)
        {
          ++sourcedone;
          if (result.complete && result.target && result.matchedlocalfilename == result.localfilename)
            ++sourcecomplete;
        }

        Verdict(result);
      }
      break;
    case phWriting:
      writtenbytes += result.filesize;
      writtenblocks += result.blocksfound;
      break;
    default:
      break;
    }
  }

  void OnError(const Par2Error &error) override
  {
    std::lock_guard<std::mutex> lock(mutex);

    if (error.code == ecDuplicateSourceFile)
      duplicates.insert(error.filename);

    Write(serr, nlSilent, Said(error) + "\n");
  }

  void OnWarning(const Par2Warning &warning) override
  {
    switch (warning.code)
    {
    case wcFilenameUnsafe:
      // A name which climbs out of the directory is said even when quiet
      Err(std::string::npos == warning.filename.find("../") ? nlNormal : nlQuiet,
          "WARNING: " + warning.message + "\n" + warning.detail);
      break;
    case wcFilenameChanged:
      Err(nlQuiet, "INFO: " + warning.message + "\n" + warning.detail);
      break;
    case wcIncompleteWrite:
      Err(nlSilent, "INFO: " + warning.message + "\n" + warning.detail);
      break;
    default:
      Err(nlSilent, warning.message + "\n" + warning.detail);
      break;
    }
  }

  void OnDetail(Verbosity verbosity, const std::string &text) override
  {
    std::lock_guard<std::mutex> lock(mutex);

    // Building or solving the matrix is done once it reaches its end
    if (started && 1000 == reached && (phase == phConstructing || phase == phSolving))
      Leave();

    // Said while a file is being read, the progress of the step is printed
    // again after it
    if (open > 0)
    {
      for (const auto &name : opening)
        Write(sout, nlNormal, "Opening: \"" + Shortened(name) + "\"\n");
      opening.clear();

      const std::string label = Label(phase);
      sout << std::string(label.size() + 6, ' ') << '\r' << text << '\n'
           << label << reached/10 << '.' << reached%10 << "%\r" << std::flush;
    }
    else
    {
      sout << text << std::endl;
    }
  }

private:
  void Write(std::ostream &stream, const NoiseLevel least, const std::string &text)
  {
    if (Shows(least) && !text.empty())
      stream << text << std::flush;
  }

  // Move on to a step of the work, saying what a step prints at its start
  void Enter(const Phase _phase)
  {
    if (started && phase == _phase)
      return;

    Leave();

    started = true;
    phase = _phase;
    reached = 0;

    switch (phase)
    {
    case phHashing:
      Write(sout, nlNormal, createsummary);
      break;
    case phConstructing:
      Write(sout, nlNormal, "Computing Reed Solomon matrix.\n");
      break;
    case phProcessing:
      if (repairing)
        Write(sout, nlQuiet, "\n");
      break;
    case phVerifyingRepair:
      Write(sout, nlQuiet, "\nVerifying repaired files:\n\n");
      break;
    case phWriting:
      writtenbytes = 0;
      writtenblocks = 0;
      break;
    default:
      break;
    }
  }

  // Say what a step prints at its end
  void Leave(void)
  {
    if (!started)
      return;

    started = false;

    switch (phase)
    {
    case phConstructing:
      Write(sout, nlNormal, "Constructing: done.\n");
      break;
    case phSolving:
      Write(sout, nlNormal, "Solving: done.\n");
      break;
    case phWriting:
      if (0 == blocksize)
      {
        Write(sout, nlNormal, "Writing recovered data\rWrote " + std::to_string(writtenbytes) + " bytes to disk\n");
      }
      else
      {
        if (writtenblocks > 0)
          Write(sout, nlNormal, "Writing recovery packets\rWrote " + std::to_string(writtenblocks * blocksize)
                                + " bytes to disk\nWriting recovery packets\n");
        Write(sout, nlNormal, "Writing verification packets\n");
      }
      break;
    default:
      break;
    }
  }

  // What the progress of a step is printed after
  const char *Label(const Phase _phase) const
  {
    switch (_phase)
    {
    case phLoading:         return "Loading: ";
    case phHashing:         return "";
    case phScanning:        return "Scanning: ";
    case phConstructing:    return "Constructing: ";
    case phSolving:         return "Solving: ";
    case phProcessing:      return repairing ? "Repairing: " : "Processing: ";
    case phVerifyingRepair: return "Scanning: ";
    case phWriting:         return "";
    }

    return "";
  }

  // A name of more than 56 characters, with the middle left out
  static std::string Shortened(const std::string &name)
  {
    if (name.size() > 56)
      return name.substr(0, 28) + "..." + name.substr(name.size() - 28);

    return name;
  }

  // A path relative to the basepath
  std::string Relative(const std::string &path) const
  {
    return DiskFile::SplitRelativeFilename(path, basepath);
  }

  // Say what checking a file found
  void Verdict(const Par2FileResult &result)
  {
    if (!Shows(nlQuiet))
      return;

    const std::string name = Relative(result.localfilename);
    const char *const kind = result.target ? "Target: \"" : "File: \"";

    std::ostringstream line;

    if (!result.exists)
    {
      if (result.target)
        line << "Target: \"" << name << "\" - missing.\n";
    }
    else if (result.scanned && 0 == result.filesize)
    {
      line << kind << name << "\" - empty.\n";
    }
    else if (result.scanned && 0 == result.blocksfound)
    {
      if (result.duplicateblocks > 0)
        line << "File: \"" << name << "\" - found " << result.duplicateblocks << " duplicate data blocks.\n";
      else
        line << "File: \"" << name << "\" - no data found.\n";

      Skipped(line, result);
    }
    else if (result.scanned)
    {
      // Whether the blocks found belong to the file at this name
      const bool own = result.target && result.matchedlocalfilename == result.localfilename;
      const std::string matched = Relative(result.matchedlocalfilename);

      if (result.complete)
      {
        if (own)
          line << "Target: \"" << name << "\" - found.\n";
        else
          line << kind << name << "\" - is a match for \"" << matched << "\".\n";
      }
      else
      {
        if (result.severalfiles)
        {
          if (result.target)
            line << "Target: \"" << name << "\" - damaged, found ";
          else
            line << "File: \"" << name << "\" - found ";
          line << result.blocksfound << " data blocks from several target files.\n";
        }
        else if (own)
        {
          line << "Target: \"" << name << "\" - damaged. Found " << result.blocksfound
               << " of " << result.blocksneeded << " data blocks.\n";
        }
        else
        {
          if (result.target)
            line << "Target: \"" << name << "\" - damaged. Found ";
          else
            line << "File: \"" << name << "\" - found ";
          line << result.blocksfound << " of " << result.blocksneeded
               << " data blocks from \"" << matched << "\".\n";
        }

        Skipped(line, result);
      }
    }

    // Matched whole by its hashes rather than block by block
    if (result.complete && 0 == result.blocksfound)
      line << result.localfilename << " is a perfect match for " << result.matchedfilename << "\n";

    Write(sout, nlQuiet, line.str());
  }

  // What the tool has always said of an error, which the library words for an
  // application
  std::string Said(const Par2Error &error) const
  {
    switch (error.code)
    {
    case ecMainPacketMissing:
      return "Main packet not found.";
    case ecDuplicateSourceFile:
      return "Source file " + Relative(error.filename) + " is a duplicate.";
    case ecFileDescriptionMissing:
      return error.message + ".\nRecovery will not be possible.";
    case ecTooManySourceBlocks:
      if (error.message == "Too many source blocks in the recovery set")
        return "Too many source blocks in recovery set.";
      break;
    case ecNotVerified:
      if (!error.filename.empty())
        return "\"" + error.filename + "\" already exists but was not scanned.";
      break;
    case ecOutOfMemory:
      if (error.message == "Memory ran out")
        return "Memory ran out.";
      return "Could not allocate buffer memory.";
    case ecProcessorFailed:
      if (error.message == "The processor the application supplied built nothing")
        return "Could not allocate buffer memory.";
      if (error.message == "The processor could not return the rebuilt data")
        return "Could not read the repaired data back from the processor.";
      if (error.message == "The processor could not return the recovery data")
        return "Could not read the recovery data back from the processor.";
      break;
    case ecInvalidSetting:
      if (error.message == "The recovery blocks would need exponents above 65535")
        return "First recovery block number is too high.";
      if (error.message == "The block size was zero")
        return "ERROR: Block size was zero!";
      if (error.message == "The block size was not a multiple of 4 bytes")
        return "ERROR: Block size was not a multiple of 4 bytes!";
      break;
    case ecFileCreateFailed:
      if (error.message == "The name is longer than this system allows")
        return error.filename + " pathlength is more than " + std::to_string(_MAX_PATH) + ".";
      break;
    default:
      break;
    }

    std::string said = error.message;
    if (!error.filename.empty() && std::string::npos == said.find(error.filename))
      said += ": " + error.filename;
    return said;
  }

  static void Skipped(std::ostringstream &line, const Par2FileResult &result)
  {
    if (result.skippedbytes > 0)
      line << result.skippedbytes << " bytes of data were skipped whilst scanning.\n"
              "If there are not enough blocks found to repair: try again with the -N option.\n";
  }

  std::ostream &sout;
  std::ostream &serr;
  const NoiseLevel noiselevel;

  std::mutex mutex;                 // Held while anything is printed
  bool started;                     // Whether a step is in progress
  Phase phase;                      // and which one
  std::string basepath;
  bool repairing;
  size_t sourcefilecount;
  size_t scanned;                   // Files a verify has started on
  bool keeploaded;
  std::vector<std::string> loaded;  // The PAR2 files read, in order
  std::string createsummary;
  u64 blocksize;                    // A create's, and 0 for a repair
  u32 reached;                      // How far the step in progress has got
  std::vector<std::string> opening; // Files started on, not yet said to be opened
  size_t open;                      // Files started on and not yet done
  size_t sourcedone;                // Files of the set a verify has checked
  size_t sourcecomplete;            // and those it found whole where they belong
  bool extraheader;                 // Whether the extra files have been announced
  std::set<std::string> duplicates; // Files the set names more than once
  u64 writtenbytes;                 // What the files written came to
  u32 writtenblocks;                // and the recovery blocks they held
};

// Delete a file, saying so when it cannot be
static void Remove(Printer &printer, const std::string &filename)
{
  DiskFile diskfile;

  if (!diskfile.Open(filename))
    return;

  printer.Out(nlQuiet, "Remove \"" + NameOf(filename) + "\".\n");

  diskfile.Close();
  if (!diskfile.Delete())
    printer.Err(nlSilent, "Cannot delete " + filename + "\n");
}

// The tool's create
static Result Create(CommandLine &commandline, const Backends &backends, Printer &printer)
{
  Par2Creator creator(commandline.GetBasePath(), backends);
  creator.SetObserver(&printer);
  creator.SetSourceFiles(commandline.GetExtraFiles());
  creator.SetBlockSize(commandline.GetBlockSize());
  creator.SetRecoveryBlockCount(commandline.GetRecoveryBlockCount());
  creator.SetRecoveryFileScheme(commandline.GetRecoveryFileScheme(), commandline.GetRecoveryFileCount());
  creator.SetFirstRecoveryBlock(commandline.GetFirstRecoveryBlock());
  creator.SetMemoryLimit(commandline.GetMemoryLimit());
  creator.SetThreadCounts(commandline.GetNumThreads(), commandline.GetFileThreads());
  creator.SetVerbosity(VerbosityAt(commandline.GetNoiseLevel()));

  u32 recoveryfilecount = commandline.GetRecoveryFileCount();
  ComputeRecoveryFileCount(0,
                           &recoveryfilecount,
                           commandline.GetRecoveryFileScheme(),
                           commandline.GetRecoveryBlockCount(),
                           commandline.GetLargestFileSize(),
                           commandline.GetBlockSize());

  std::ostringstream summary;
  summary << "Block size: " << commandline.GetBlockSize() << "\n"
    "Source file count: " << commandline.GetExtraFiles().size() << "\n"
    "Source block count: " << commandline.GetSourceBlockCount() << "\n"
    "Recovery block count: " << commandline.GetRecoveryBlockCount() << "\n"
    "Recovery file count: " << recoveryfilecount << "\n"
    "\n";
  printer.SetCreateSummary(summary.str());
  printer.SetBlockSize(commandline.GetBlockSize());

  const Result result = creator.Create(commandline.GetParFilename() + ".par2");
  printer.Finish();

  if (result == eSuccess)
    printer.Out(nlQuiet, "Done\n");

  return result;
}

// The tool's verify, and repair when it was asked for
static Result Repair(CommandLine &commandline, const Backends &backends, Printer &printer)
{
  Par2Verifier verifier(commandline.GetBasePath(), backends);
  verifier.SetObserver(&printer);
  verifier.SetMemoryLimit(commandline.GetMemoryLimit());
  verifier.SetThreadCounts(commandline.GetNumThreads(), commandline.GetFileThreads());
  verifier.SetDataSkipping(commandline.GetSkipData(), commandline.GetSkipLeaway());
  verifier.SetFullHash(commandline.GetFullHash());
  verifier.SetRenameOnly(commandline.GetRenameOnly());
  verifier.SetVerbosity(VerbosityAt(commandline.GetNoiseLevel()));

  printer.SetBasePath(NormaliseBasePath(commandline.GetBasePath()));

  // The PAR2 files read for the set named are the ones a purge deletes, and
  // those named among the extra files are only read
  printer.KeepLoaded(true);
  Result result = verifier.AddPar2File(commandline.GetParFilename());
  printer.KeepLoaded(false);

  for (const auto &extrafile : commandline.GetExtraFiles())
  {
    if (!Par2Repairer::IsPar2Filename(extrafile))
      continue;

    const Result added = verifier.AddPar2File(extrafile, false);
    if (added != eFileIOError)
      result = added;
  }

  // Nothing could be read, so nothing describes the set
  if (result == eFileIOError)
    result = eInsufficientCriticalData;

  printer.Out(nlNormal, "\n");

  if (result != eSuccess)
    return result;

  Par2SetInfo info;
  verifier.GetSetInfo(&info);

  {
    std::ostringstream text;
    text << "There are " << info.recoverablefilecount << " recoverable files and "
         << info.otherfilecount << " other files.\n"
            "The block size used was " << info.blocksize << " bytes.\n"
            "There are a total of " << info.datablockcount << " data blocks.\n"
            "The total size of the data files is " << info.datasize << " bytes.\n";
    printer.Out(nlNormal, text.str());
  }

  std::vector<Par2FileInfo> files;
  verifier.GetFileInfo(&files);
  printer.SetSourceFileCount(files.size());

  printer.Out(nlNormal, "\nVerifying source files:\n\n");

  result = verifier.Verify(commandline.GetExtraFiles());
  printer.Finish();

  Par2VerifyResult verified;
  if (!verifier.GetVerifyResult(&verified))
    return result;

  printer.Verified(info.recoverablefilecount);

  printer.Out(nlQuiet, "\n");

  if (verified.completefilecount < info.recoverablefilecount ||
      verified.renamedfilecount > 0 ||
      verified.damagedfilecount > 0 ||
      verified.missingfilecount > 0)
  {
    printer.Out(nlQuiet, "Repair is required.\n");

    std::ostringstream summary;
    if (verified.renamedfilecount > 0) summary << verified.renamedfilecount << " file(s) have the wrong name.\n";
    if (verified.missingfilecount > 0) summary << verified.missingfilecount << " file(s) are missing.\n";
    if (verified.damagedfilecount > 0) summary << verified.damagedfilecount << " file(s) exist but are damaged.\n";
    if (verified.completefilecount > 0) summary << verified.completefilecount << " file(s) are ok.\n";

    summary << "You have " << verified.availableblockcount
            << " out of " << info.datablockcount
            << " data blocks available.\n";
    if (verified.recoveryblockcount > 0)
      summary << "You have " << verified.recoveryblockcount
              << " recovery blocks available.\n";
    printer.Out(nlNormal, summary.str());

    if (verified.recoveryblockcount >= verified.missingblockcount)
    {
      printer.Out(nlQuiet, "Repair is possible.\n");

      std::ostringstream usage;
      if (verified.recoveryblockcount > verified.missingblockcount)
        usage << "You have an excess of "
              << verified.recoveryblockcount - verified.missingblockcount
              << " recovery blocks.\n";

      if (verified.missingblockcount > 0)
        usage << verified.missingblockcount
              << " recovery blocks will be used to repair.\n";
      else if (verified.recoveryblockcount > 0)
        usage << "None of the recovery blocks will be used for the repair.\n";
      printer.Out(nlNormal, usage.str());
    }
    else
    {
      std::ostringstream needed;
      needed << "Repair is not possible.\n"
                "You need " << verified.missingblockcount - verified.recoveryblockcount
             << " more recovery blocks to be able to repair.\n";
      printer.Out(nlQuiet, needed.str());
    }
  }
  else
  {
    printer.Out(nlQuiet, "All files are correct, repair is not required.\n");
  }

  if (result == eRepairPossible)
  {
    if (commandline.GetOperation() != CommandLine::opRepair)
      return result;

    printer.SetRepairing(verified.missingblockcount > 0);
    printer.Out(nlQuiet, "\n");

    result = verifier.Repair();
    printer.Finish();

    if (result == eRepairFailed)
      printer.Err(nlSilent, "Repair Failed.\n");

    if (result != eSuccess)
      return result;

    printer.Out(nlQuiet, "\nRepair complete.\n");
  }
  else if (result != eSuccess)
  {
    return result;
  }

  if (commandline.GetPurgeFiles())
  {
    const std::vector<std::string> backups = verifier.GetBackupFiles();

    if (!backups.empty())
      printer.Out(nlQuiet, "\nPurge backup files.\n");

    for (const auto &backup : backups)
      Remove(printer, backup);

    printer.Out(nlQuiet, "\nPurge par files.\n");

    for (const auto &par2file : printer.Loaded())
      Remove(printer, par2file);
  }

  return eSuccess;
}

Result run(int argc, const char * const *argv, std::ostream &sout, std::ostream &serr,
           const Backends &backends)
try
{
  // Parse the command line
  CommandLine commandline(sout, serr);

  Result result = eInvalidCommandLineArguments;

  if (commandline.Parse(argc, argv))
  {
    // Prints what the library does
    Printer printer(sout, serr, commandline.GetNoiseLevel());

    // Which operation was selected
    switch (commandline.GetOperation())
    {
      case CommandLine::opCreate:
        // Create recovery data
        result = Create(commandline, backends, printer);
        break;
      case CommandLine::opVerify:
      case CommandLine::opRepair:
        {
          // Verify or Repair damaged files
          switch (commandline.GetVersion())
          {
            case CommandLine::verPar1:
              printer.SetRepairing(true);
              result = par1repair(sout,
                                  serr,
                                  commandline.GetNoiseLevel(),
                                  commandline.GetMemoryLimit(),
                                  commandline.GetNumThreads(),
                                  commandline.GetParFilename(),
                                  commandline.GetExtraFiles(),
                                  commandline.GetOperation() == CommandLine::opRepair,
                                  commandline.GetPurgeFiles(),
                                  &printer);
              printer.Finish();
              break;
            case CommandLine::verPar2:
              result = Repair(commandline, backends, printer);
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

  return result;
}
catch (...)
{
  return Thrown(serr);
}

#ifdef _WIN32

Result run(int argc, wchar_t *wargv[], std::ostream &sout, std::ostream &serr,
           const Backends &backends)
try
{
  utf8::WideToUtf8ArgsAdapter wargsAdapter{ argc, wargv, serr };

  return run(wargsAdapter.GetArgc(), wargsAdapter.GetUtf8Args(), sout, serr, backends);
}
catch (...)
{
  return Thrown(serr);
}

#endif

} // namespace par2
