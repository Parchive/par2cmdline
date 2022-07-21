#include "libpar2.h"
#include "galois_cu.cuh"
#include <stdio.h>

__global__
void callDevice(GaloisCu16 *a, GaloisCu16 *b)
{
  printf("On device\n");
  printf("a + b is %u\n", (*a + *b).Value());
  printf("a - b is %u\n", (*a - *b).Value());
  printf("a * b is %u\n", (*a * *b).Value());
}

int main()
{
  GaloisCu16::uploadTable();
  Galois<16,0x1100B,u_int16_t> a(23);
  Galois<16,0x1100B,u_int16_t> b(532);
  printf("a + b is %u\n", (a + b).Value());
  printf("a - b is %u\n", (a - b).Value());
  printf("a * b is %u\n", (a * b).Value());

  GaloisCu16 *da, *db;
  cudaMalloc((void**) &da, sizeof(GaloisCu16));
  cudaMalloc((void**) &db, sizeof(GaloisCu16));
  cudaMemcpy(da, &a, sizeof(GaloisCu16), cudaMemcpyHostToDevice);
  cudaMemcpy(db, &b, sizeof(GaloisCu16), cudaMemcpyHostToDevice);
  callDevice<<<1, 1>>> (da, db);
  cudaDeviceSynchronize();

  return 0;
}