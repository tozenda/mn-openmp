#include "mnblas.h"

void mncblas_sswap(const int N, float *X, const int incX, float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register float save ;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
  {
    save = Y [j] ;
    Y [j] = X [i] ;
    X [i] = save ;
  }

  return ;
}

void mncblas_sswap_noomp(const int N, float *X, const int incX, float *Y, const int incY)
{
  register unsigned int i;
  register float save ;

  for (i=0;i < N;i += incX)
  {
    save = Y [i] ;
    Y [i] = X [i] ;
    X [i] = save ;
  }
}

void mncblas_sswap_omp(const int N, float *X, const int incX, float *Y, const int incY)
{
  register unsigned int i;
  register float save ;
  #pragma omp for schedule(static)
  for (i=0;i < N;i += incX)
  {
    save = Y [i] ;
    Y [i] = X [i] ;
    X [i] = save ;
  }
}

void mncblas_dswap(const int N, double *X, const int incX,double *Y, const int incY)
{

  return ;
}

void mncblas_cswap(const int N, void *X, const int incX,void *Y, const int incY)
{

  return ;
}

void mncblas_zswap(const int N, void *X, const int incX,void *Y, const int incY)
{

  return ;
}
