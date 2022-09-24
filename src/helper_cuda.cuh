#pragma once
#include <stdio.h>

#define cudaErrchk(ans) { if ( !gpuAssert((ans), __FILE__, __LINE__) ) return false; }
inline bool gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      // if (abort) exit(code);
      return false;
   }
   return true;
}