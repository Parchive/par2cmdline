#include <assert.h>

#include "libpar2internal.h"
#include "helper_cuda.cuh"

template <typename G>
__global__ void ProcessKer( const int chunkSz, const int inputCount,
                            const void * __restrict__ inputBuf,
                            const void * __restrict__ bases,
                            const int outputCount,
                            void * __restrict__ outputBuf,
                            const void * __restrict__ exponents,
                            const int sharedMemSz );

// Calculate a chunk of all recovery blocks on CUDA device
template<typename g>
bool ReedSolomon<g>::ProcessCu(size_t size, u32 inputIdxStart, u32 inputIdxEnd, const void *inputBuf,
                               u32 outputIdxStart, u32 outputIdxEnd, void *outputBuf)
{
  // CUDA Device compatible Galois type.
  typedef GaloisCu<G::Bits, G::Generator, G::ValueType> Gd;
  cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
  u32 outCount = outputIdxEnd - outputIdxStart + 1;
  /* 
  * size: chunk size
  * VRam footprint: (inputcount + outCount) * size + (inputcount + outCount) * sizeof(g) + sizeof(g::GaloisTable)
  *                 <input buffer and output buffer>          <base and exponent>            <Galois log tables>
  * 
  * Align sharedBufSz to 4 bytes
  * Align chunkSz to 4 bytes
  * 
  */
  
}

template <typename G>
__global__ void ProcessKer( const int chunkSz, const int inputCount,
                            const void * __restrict__ inputBuf,
                            const void * __restrict__ bases,
                            const int outputCount,
                            void * __restrict__ outputBuf,
                            const void * __restrict__ exponents,
                            const int sharedBufSz )
{
  // G can only be GF(8) or GF(16) or GF(32)
  // inputBuf should be size chunkSz * inputCount.
  // outputBuf should be size chunkSz * outputCount.
  // Each thread block compute an output block.
  // Each warp compute the contribution of an input block.
  // -- Coalesced read from global inputBuf, store RS entry in register.
  // Output is buffered in shared memory.
  // Results from each warp are accumulated in shared memory by atomic XOR.
  // Shared memory output buffer is written to global memory outputBuf after
  // contribution from all input blocks are computed and accumulated. (syncthreads)

  // Grid size should match output count 
  assert( blockDim.x == outputCount );

  int outputIdx = blockIdx.x;
  int warpCount = blockDim.x / 32;    // # of warps in a block
  int warpIdx = threadIdx.x / 32;     // which warp is this thread in
  int intCount = chunkSz / 4;         // # of ints in a chunk
  int wordPerInt = 32 / G::Bits;      // # of words in an int
  int tIdxWarp = threadIdx.x % 32;    // Index of thread in its warp

  // Declare shared mem output buffer and initialize to 0
  __shared__ u8 outSharedBuf[sharedBufSz];
  for ( int i = threadIdx.x; i < sharedBufSz / 4; i += blockDim.x )
    ((u32*) outSharedBuf)[i] = 0;

  // For each input block
  for ( int inputIdx = warpIdx; i < inputCount; inputIdx += warpCount )
  {
    // Calculate RS matrix entry for the input block
    G left = ((G*) bases)[inputIdx].pow( ((G*) exponents)[outputIdx] );
    
    #pragma unroll
    for ( int i = tIdxWarp; i < intCount; i += 32 )
    {
      // Load words as an int.
      u32 words = ((u32*) inputBuf)[inputIdx * intCount + i];
      u32 res;
      for ( int j = 0; j < wordPerInt; ++j )
      {
        ((G*)&res)[j] = left * ((G*)&words)[j];
      }
      // TODO: Compare atomicXor, atomicXor_block perf.
      // TODO: Figure out outputSharedBuf indexing
      atomicXor( outSharedBuf[...], res );

    }

  }


}