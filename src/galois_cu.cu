#include "libpar2internal.h"

// template <const unsigned int bits, const unsigned int generator, typename valuetype>
// __device__ GaloisTable<bits,generator,valuetype> *d_table;

template<>
bool GaloisCu16::uploaded = false;