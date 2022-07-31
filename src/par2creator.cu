#include "libpar2internal.h"
#include "helper_cuda.cuh"

#ifdef _MSC_VER
#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif
#endif

// Allocate pinned (page locked) memory buffers for reading and writing data to disk.
// This enables faster data transfer between host and CUDA device.
bool Par2Creator::AllocateBuffersPinned(void)
{
  cudaError_t err = cudaSuccess;
  err = cudaMallocHost( ( void** ) &inputbuffer, chunksize );

  // If pinned mem allocation failed, fall back to pageable memory.
  if ( err != cudaSuccess )
    return AllocateBuffers();
  
  err = cudaMallocHost( ( void** ) &outputbuffer, chunksize * recoveryblockcount );

  // If pinned mem allocation failed, fall back to pageable memory.
  if ( err != cudaSuccess )
    return AllocateBuffers();

  return true;
}

// ProcessData, but on CUDA device.
bool Par2Creator::ProcessDataCu(u64 blockoffset, size_t blocklength)
{
  // Clear the output buffer
  memset(outputbuffer, 0, chunksize * recoveryblockcount);

  // If we have deferred computation of the file hash and block crc and hashes
  // sourcefile and sourceindex will be used to update them during
  // the main recovery block computation
  vector<Par2CreatorSourceFile*>::iterator sourcefile = sourcefiles.begin();
  u32 sourceindex = 0;

  vector<DataBlock>::iterator sourceblock;
  u32 inputblock;

  DiskFile *lastopenfile = NULL;

  // For each input block
  for ((sourceblock=sourceblocks.begin()),(inputblock=0);
       sourceblock != sourceblocks.end();
       ++sourceblock, ++inputblock)
  {
    // Are we reading from a new file?
    if (lastopenfile != (*sourceblock).GetDiskFile())
    {
      // Close the last file
      if (lastopenfile != NULL)
      {
        lastopenfile->Close();
      }

      // Open the new file
      lastopenfile = (*sourceblock).GetDiskFile();
      if (!lastopenfile->Open())
      {
        return false;
      }
    }

    // Read data from the current input block
    if (!sourceblock->ReadData(blockoffset, blocklength, inputbuffer))
      return false;

    if (deferhashcomputation)
    {
      assert(blockoffset == 0 && blocklength == blocksize);
      assert(sourcefile != sourcefiles.end());

      (*sourcefile)->UpdateHashes(sourceindex, inputbuffer, blocklength);
    }


  }
}