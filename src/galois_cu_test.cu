#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <assert.h>
#include "libpar2.h"
#include "galois_cu.cuh"
#include "helper_cuda.cuh"
#include "../profiling/Timer.h"

#define BLOCK_WIDTH 64

void TestMult(void);
void TestDiv(void);
void TestAdd(void);
void TestSub(void);
void CompareResults(const Galois16 *resCPU, const GaloisCu16 *resGPU, char op);

__global__
void callDevice( GaloisCu16 *a, GaloisCu16 *b )
{
  printf( "On device\n" );

  printf( "a * b is %hu\n", ( *a * *b ).Value() );
}

int main()
{
  GaloisCu16::uploadTable();

  // u16 ua = 52235;
  // u16 ub = 65521;
  // GaloisCu16 da(ua), db(ub);
  // Galois16 a(ua), b(ub);

  // printf("%hu\n", (a * b).Value());
  // GaloisCu16 *dap, *dbp;
  // cudaMalloc((void**) &dap, 2);
  // cudaMalloc((void**) &dbp, 2);
  // cudaMemcpy(dap, &da, 2, cudaMemcpyHostToDevice);
  // cudaMemcpy(dbp, &db, 2, cudaMemcpyHostToDevice);
  // callDevice<<<1, 1>>> (dap, dbp);
  // cudaDeviceSynchronize();


  TestMult();
  printf("----------------------\n");
  TestDiv();
  printf("----------------------\n");
  TestAdd();
  printf("----------------------\n");
  TestSub();

  return 0;
}

void CompareResults(const Galois16 *resCPU, const GaloisCu16 *resGPU, char op)
{
  printf("Varifying results.\n");
  bool correct = true;
  size_t resId = 0;
  for ( size_t i = 0; i < Galois16::Count; ++i )
  {
    for ( size_t j = i; j < Galois16::Count; ++j )
    {
      if ( resCPU[resId].Value() != resGPU[resId].Value() )
      {
        correct = false;
        printf("Result doesn't match! %u %c %u = %hu, but got %hu.\n",
                i, op, j, resCPU[resId], resGPU[resId]);
      }
      ++resId;
    }
  }

  if (correct)
  {
    printf("Operation %c is correct!\n", op);
  }
}


__global__
void KerMult(__restrict__ GaloisCu16 *vars, __restrict__ GaloisCu16 *res)
{
  size_t outRowIdx = blockIdx.x * blockDim.x + threadIdx.x;   // Also the idx of left var.
  size_t outColMax = GaloisCu16::Count - outRowIdx;
  size_t outStartIdx = ( outColMax + GaloisCu16::Count + 1 ) * outRowIdx / 2;
  GaloisCu16 left = vars[ outRowIdx ];

  for ( size_t i = 0; i < outColMax; ++i )
  {
    res[ outStartIdx + i ] = left * vars[ outRowIdx + i ];
  }
  
}

void TestMult()
{
  Galois16 vals[ Galois16::Count ], *results, *resultsOmp;
  GaloisCu16 valsCu[ GaloisCu16::Count ], *resultsCu;

  // Fill arrays with every element in GF(2^16)
  for ( size_t i = 0; i < Galois16::Count; ++i )
  {
    vals[ i ] = Galois16( ( Galois16::ValueType ) i );
    valsCu[ i ] = GaloisCu16( ( Galois16::ValueType ) i );
  }
  printf("Input arrays filled.\n");

  // Allocate space for multiplication result
  size_t resultSz = sizeof( Galois16 ) * ( Galois16::Count + 1 ) * Galois16::Count / 2;
  // results = ( Galois16* ) malloc(resultSz);
  resultsOmp = ( Galois16* ) malloc(resultSz);
  cudaMallocHost( ( void** ) &resultsCu, resultSz );
  printf("Output arrays allocated\n");

  // Calculate multiplication results on CPU.
  // printf("Calculating reference results.\n");
  // {
  //   Timer timer("CPU-Serial");

  //   size_t resId = 0;
  //   for ( size_t i = 0; i < Galois16::Count; ++i )
  //   {
  //     for ( size_t j = i; j < Galois16::Count; ++j )
  //     {
  //       results[resId] = vals[i] * vals[j];
  //       ++resId;
  //     }
  //   }
  // }
  // printf("Finished calculating reference results.\n");

  // Calculate results using openmp
  printf("Calculating OMP results.\n");
  {
    Timer timer("CPU-OMP");

    #pragma omp parallel for
    for ( size_t i = 0; i < Galois16::Count; ++i )
    {
      for ( size_t j = i; j < Galois16::Count; ++j )
      {
        resultsOmp[i * ( 2 * Galois16::Count - i + 1 ) / 2 + j - i] = vals[i] * vals[j];
      }
    }
  }

  // Upload data to GPU
  GaloisCu16 *d_valsCu, *d_resultsCu;
  {
    Timer timer("GPU");
    cudaErrchk( cudaMalloc( (void**) &d_valsCu, sizeof( valsCu ) ) );
    cudaErrchk( cudaMalloc( (void**) &d_resultsCu, resultSz ) );
    cudaErrchk( cudaMemcpy( d_valsCu, valsCu, sizeof( valsCu ), cudaMemcpyHostToDevice ) );
    printf("Input copied to GPU.\n");

    // Launch kernel
    dim3 dimBlock(BLOCK_WIDTH);
    dim3 dimGrid(GaloisCu16::Count / BLOCK_WIDTH);
    KerMult<<<dimGrid, dimBlock>>> (d_valsCu, d_resultsCu);
    cudaErrchk( cudaGetLastError() );
    printf("Kernel launched.\n");

    cudaDeviceSynchronize();
    printf("Kernel completed.\n");

    // Download results from GPU
    cudaErrchk ( cudaMemcpy(resultsCu, d_resultsCu, resultSz, cudaMemcpyDeviceToHost) );
  }

  CompareResults(resultsOmp, resultsCu, '*');


  // free(results);
  free(resultsOmp);
  cudaFreeHost(resultsCu);
  cudaFree(d_resultsCu);
}

__global__
void KerDiv(__restrict__ GaloisCu16 *vars, __restrict__ GaloisCu16 *res)
{
  size_t outRowIdx = blockIdx.x * blockDim.x + threadIdx.x;   // Also the idx of left var.
  size_t outColMax = GaloisCu16::Count - outRowIdx;
  size_t outStartIdx = ( outColMax + GaloisCu16::Count + 1 ) * outRowIdx / 2;
  GaloisCu16 left = vars[ outRowIdx ];

  for ( size_t i = 0; i < outColMax; ++i )
  {
    res[ outStartIdx + i ] = left / vars[ outRowIdx + i ];
  }
  
}

void TestDiv()
{
  Galois16 vals[ Galois16::Count ], *resultsOmp;
  GaloisCu16 valsCu[ GaloisCu16::Count ], *resultsCu;

  // Fill arrays with every element in GF(2^16)
  for ( size_t i = 0; i < Galois16::Count; ++i )
  {
    vals[ i ] = Galois16( ( Galois16::ValueType ) i );
    valsCu[ i ] = GaloisCu16( ( Galois16::ValueType ) i );
  }
  printf("Input arrays filled.\n");

  // Allocate space for result
  size_t resultSz = sizeof( Galois16 ) * ( Galois16::Count + 1 ) * Galois16::Count / 2;
  resultsOmp = ( Galois16* ) malloc(resultSz);
  cudaMallocHost( ( void** ) &resultsCu, resultSz );
  printf("Output arrays allocated\n");

  // Calculate results using openmp
  printf("Calculating Reference results.\n");
  {
    Timer timer("CPU-OMP");

    #pragma omp parallel for
    for ( size_t i = 0; i < Galois16::Count; ++i )
    {
      for ( size_t j = i; j < Galois16::Count; ++j )
      {
        resultsOmp[i * ( 2 * Galois16::Count - i + 1 ) / 2 + j - i] = vals[i] / vals[j];
      }
    }
  }

  // Upload data to GPU
  GaloisCu16 *d_valsCu, *d_resultsCu;
  {
    Timer timer("GPU");
    cudaErrchk( cudaMalloc( (void**) &d_valsCu, sizeof( valsCu ) ) );
    cudaErrchk( cudaMalloc( (void**) &d_resultsCu, resultSz ) );
    cudaErrchk( cudaMemcpy( d_valsCu, valsCu, sizeof( valsCu ), cudaMemcpyHostToDevice ) );
    printf("Input copied to GPU.\n");

    // Launch kernel
    dim3 dimBlock(BLOCK_WIDTH);
    dim3 dimGrid(GaloisCu16::Count / BLOCK_WIDTH);
    KerDiv<<<dimGrid, dimBlock>>> (d_valsCu, d_resultsCu);
    cudaErrchk( cudaGetLastError() );
    printf("Kernel launched.\n");

    cudaDeviceSynchronize();
    printf("Kernel completed.\n");

    // Download results from GPU
    cudaErrchk ( cudaMemcpy(resultsCu, d_resultsCu, resultSz, cudaMemcpyDeviceToHost) );
  }

  CompareResults(resultsOmp, resultsCu, '/');

  // free(results);
  free(resultsOmp);
  cudaFreeHost(resultsCu);
  cudaFree(d_resultsCu);
}

__global__
void KerAdd(__restrict__ GaloisCu16 *vars, __restrict__ GaloisCu16 *res)
{
  size_t outRowIdx = blockIdx.x * blockDim.x + threadIdx.x;   // Also the idx of left var.
  size_t outColMax = GaloisCu16::Count - outRowIdx;
  size_t outStartIdx = ( outColMax + GaloisCu16::Count + 1 ) * outRowIdx / 2;
  GaloisCu16 left = vars[ outRowIdx ];

  for ( size_t i = 0; i < outColMax; ++i )
  {
    res[ outStartIdx + i ] = left + vars[ outRowIdx + i ];
  }
  
}

void TestAdd()
{
  Galois16 vals[ Galois16::Count ], *resultsOmp;
  GaloisCu16 valsCu[ GaloisCu16::Count ], *resultsCu;

  // Fill arrays with every element in GF(2^16)
  for ( size_t i = 0; i < Galois16::Count; ++i )
  {
    vals[ i ] = Galois16( ( Galois16::ValueType ) i );
    valsCu[ i ] = GaloisCu16( ( Galois16::ValueType ) i );
  }
  printf("Input arrays filled.\n");

  // Allocate space for result
  size_t resultSz = sizeof( Galois16 ) * ( Galois16::Count + 1 ) * Galois16::Count / 2;
  resultsOmp = ( Galois16* ) malloc(resultSz);
  cudaMallocHost( ( void** ) &resultsCu, resultSz );
  printf("Output arrays allocated\n");

  // Calculate results using openmp
  printf("Calculating Reference results.\n");
  {
    Timer timer("CPU-OMP");

    #pragma omp parallel for
    for ( size_t i = 0; i < Galois16::Count; ++i )
    {
      for ( size_t j = i; j < Galois16::Count; ++j )
      {
        resultsOmp[i * ( 2 * Galois16::Count - i + 1 ) / 2 + j - i] = vals[i] + vals[j];
      }
    }
  }

  // Upload data to GPU
  GaloisCu16 *d_valsCu, *d_resultsCu;
  {
    Timer timer("GPU");
    cudaErrchk( cudaMalloc( (void**) &d_valsCu, sizeof( valsCu ) ) );
    cudaErrchk( cudaMalloc( (void**) &d_resultsCu, resultSz ) );
    cudaErrchk( cudaMemcpy( d_valsCu, valsCu, sizeof( valsCu ), cudaMemcpyHostToDevice ) );
    printf("Input copied to GPU.\n");

    // Launch kernel
    dim3 dimBlock(BLOCK_WIDTH);
    dim3 dimGrid(GaloisCu16::Count / BLOCK_WIDTH);
    KerAdd<<<dimGrid, dimBlock>>> (d_valsCu, d_resultsCu);
    cudaErrchk( cudaGetLastError() );
    printf("Kernel launched.\n");

    cudaDeviceSynchronize();
    printf("Kernel completed.\n");

    // Download results from GPU
    cudaErrchk ( cudaMemcpy(resultsCu, d_resultsCu, resultSz, cudaMemcpyDeviceToHost) );
  }

  CompareResults(resultsOmp, resultsCu, '+');

  // free(results);
  free(resultsOmp);
  cudaFreeHost(resultsCu);
  cudaFree(d_resultsCu);
}

__global__
void KerSub(__restrict__ GaloisCu16 *vars, __restrict__ GaloisCu16 *res)
{
  size_t outRowIdx = blockIdx.x * blockDim.x + threadIdx.x;   // Also the idx of left var.
  size_t outColMax = GaloisCu16::Count - outRowIdx;
  size_t outStartIdx = ( outColMax + GaloisCu16::Count + 1 ) * outRowIdx / 2;
  GaloisCu16 left = vars[ outRowIdx ];

  for ( size_t i = 0; i < outColMax; ++i )
  {
    res[ outStartIdx + i ] = left - vars[ outRowIdx + i ];
  }
  
}

void TestSub()
{
  Galois16 vals[ Galois16::Count ], *resultsOmp;
  GaloisCu16 valsCu[ GaloisCu16::Count ], *resultsCu;

  // Fill arrays with every element in GF(2^16)
  for ( size_t i = 0; i < Galois16::Count; ++i )
  {
    vals[ i ] = Galois16( ( Galois16::ValueType ) i );
    valsCu[ i ] = GaloisCu16( ( Galois16::ValueType ) i );
  }
  printf("Input arrays filled.\n");

  // Allocate space for result
  size_t resultSz = sizeof( Galois16 ) * ( Galois16::Count + 1 ) * Galois16::Count / 2;
  resultsOmp = ( Galois16* ) malloc(resultSz);
  cudaMallocHost( ( void** ) &resultsCu, resultSz );
  printf("Output arrays allocated\n");

  // Calculate results using openmp
  printf("Calculating Reference results.\n");
  {
    Timer timer("CPU-OMP");

    #pragma omp parallel for
    for ( size_t i = 0; i < Galois16::Count; ++i )
    {
      for ( size_t j = i; j < Galois16::Count; ++j )
      {
        resultsOmp[i * ( 2 * Galois16::Count - i + 1 ) / 2 + j - i] = vals[i] - vals[j];
      }
    }
  }

  // Upload data to GPU
  GaloisCu16 *d_valsCu, *d_resultsCu;
  {
    Timer timer("GPU");
    cudaErrchk( cudaMalloc( (void**) &d_valsCu, sizeof( valsCu ) ) );
    cudaErrchk( cudaMalloc( (void**) &d_resultsCu, resultSz ) );
    cudaErrchk( cudaMemcpy( d_valsCu, valsCu, sizeof( valsCu ), cudaMemcpyHostToDevice ) );
    printf("Input copied to GPU.\n");

    // Launch kernel
    dim3 dimBlock(BLOCK_WIDTH);
    dim3 dimGrid(GaloisCu16::Count / BLOCK_WIDTH);
    KerSub<<<dimGrid, dimBlock>>> (d_valsCu, d_resultsCu);
    cudaErrchk( cudaGetLastError() );
    printf("Kernel launched.\n");

    cudaDeviceSynchronize();
    printf("Kernel completed.\n");

    // Download results from GPU
    cudaErrchk ( cudaMemcpy(resultsCu, d_resultsCu, resultSz, cudaMemcpyDeviceToHost) );
  }

  CompareResults(resultsOmp, resultsCu, '-');

  // free(results);
  free(resultsOmp);
  cudaFreeHost(resultsCu);
  cudaFree(d_resultsCu);
}