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
  #pragma omp for schedule(static) private (save)
  for (i=0;i < N;i += incX)
  {
    save = Y [i] ;
    Y [i] = X [i] ;
    X [i] = save ;
  }
}

void mncblas_dswap_noomp(const int N, double *X, const int incX,double *Y, const int incY)
{
  register unsigned int i;
  register double save ;
  for (i=0;i < N;i += incX)
  {
    save = Y [i] ;
    Y [i] = X [i] ;
    X [i] = save ;
  }
}

void mncblas_dswap_omp(const int N, double *X, const int incX,double *Y, const int incY)
{
  register unsigned int i;
  register double save ;
  #pragma omp parallel for schedule (static) private(save)
  for (i=0;i < N;i += incX)
  {
    save = Y [i] ;
    Y [i] = X [i] ;
    X [i] = save ;
  }
}

void mncblas_cswap_noomp(const int N, void *X, const int incX,void *Y, const int incY)
{
  register unsigned int i;
  register float save ;

  float *x = (float *)X;
  float *y = (float *)Y;

  for (i=0;i < N;i += incX)
  {
    save = y [i] ;
    y [i] = x [i] ;
    x [i] = save ;
  }
}

void mncblas_cswap_omp(const int N, void *X, const int incX,void *Y, const int incY)
{
  register unsigned int i;
  register float save ;

  float *x = (float *)X;
  float *y = (float *)Y;

  #pragma omp for schedule(static) private (save)
  for (i=0;i < N;i += incX)
  {
    save = y [i] ;
    y [i] = x [i] ;
    x [i] = save ;
  }
}

void mncblas_zswap_noomp(const int N, void *X, const int incX,void *Y, const int incY)
{
  register unsigned int i;
  register double save ;

  double *x = (double *)X;
  double *y = (double *)Y;

  for (i=0;i < N;i += incX)
  {
    save = y [i] ;
    y [i] = x [i] ;
    x [i] = save ;
  }
}

void mncblas_zswap_omp(const int N, void *X, const int incX,void *Y, const int incY)
{
  register unsigned int i;
  register double save ;

  double *x = (double *)X;
  double *y = (double *)Y;

  #pragma omp for schedule(static) private (save)
  for (i=0;i < N;i += incX)
  {
    save = y [i] ;
    y [i] = x [i] ;
    x [i] = save ;
  }
}