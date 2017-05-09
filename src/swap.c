// COMPARAISON PERF : DOUBLE (d)

#include "mnblas.h"

#include <stdio.h>
#include <x86intrin.h>
#include <nmmintrin.h>


/*************************** SSWAP **************************/

void mncblas_sswap_vec(const int N, float *X, const int incX, float *Y, const int incY)
{
  register unsigned int i;

  __m128 x, y;
  int incx, incy;

  #pragma omp for schedule(static) private(incx, incy)
  for (i=0;i < N;i += 4)
  {
    incx = i*incX;
    incy = i*incY;

    x = _mm_load_ps (X+incx) ;
    y = _mm_load_ps (Y+incy) ;
    _mm_store_ps (Y+incy, x) ;
    _mm_store_ps (X+incx, y) ;
  }
}

void mncblas_sswap_noomp(const int N, float *X, const int incX, float *Y, const int incY)
{
  register unsigned int i;
  register float save ;

  int incx, incy;

  for (i=0;i < N;i += 1)
  {
    incx = i*incX;
    incy = i*incY;

    save = Y [incy] ;
    Y [incy] = X [incx] ;
    X [incx] = save ;
  }
}

void mncblas_sswap_omp(const int N, float *X, const int incX, float *Y, const int incY)
{
  register unsigned int i;
  register float save ;
  int incx, incy;

  #pragma omp for schedule(static) private(save, incx, incy)
  for (i=0;i < N;i += 1)
  {
    incx = i*incX;
    incy = i*incY;

    save = Y [incy] ;
    Y [incy] = X [incx] ;
    X [incx] = save ;
  }
}

/*************************** DSWAP **************************/

void mncblas_dswap_vec(const int N, double *X, const int incX, double *Y, const int incY)
{
  register unsigned int i;

  __m128d x, y;
  int incx, incy;

  #pragma omp for schedule(static) private(incx, incy)
  for (i=0;i < N;i += 2)
  {
    incx = i*incX;
    incy = i*incY;
    x = _mm_load_pd (X+incx) ;
    y = _mm_load_pd (Y+incy) ;
    _mm_store_pd (Y+incy, x) ;
    _mm_store_pd (X+incx, y) ;
  }
}

void mncblas_dswap_noomp(const int N, double *X, const int incX,double *Y, const int incY)
{
  register unsigned int i;
  register double save ;
  int incx, incy;

  for (i=0;i < N;i += 1)
  {
    incx = i*incX;
    incy = i*incY;

    save = Y [incy] ;
    Y [incy] = X [incx] ;
    X [incx] = save ;
  }
}

void mncblas_dswap_omp(const int N, double *X, const int incX,double *Y, const int incY)
{
  register unsigned int i;
  register double save ;
  int incx, incy;

  #pragma omp parallel for schedule (static) private(save, incx, incy)
  for (i=0;i < N;i += 1)
  {
    incx = i*incX;
    incy = i*incY;

    save = Y [incy] ;
    Y [incy] = X [incx] ;
    X [incx] = save ;
  }
}

/*************************** CSWAP **************************/

void mncblas_cswap_vec(const int N, void *X, const int incX, void *Y, const int incY)
{
  register unsigned int i;

  __m128 x, y;
  int incx, incy;

  float *a = (float *)X;
  float *b = (float *)Y;

  #pragma omp for schedule(static) private(incx, incy)
  for (i=0;i < 2*N;i += 4)
  {
    incx = i*incX;
    incy = i*incY;

    x = _mm_load_ps (a+incx) ;
    y = _mm_load_ps (b+incy) ;
    _mm_store_ps (b+incy, x) ;
    _mm_store_ps (a+incx, y) ;
  }
}

void mncblas_cswap_noomp(const int N, void *X, const int incX,void *Y, const int incY)
{
  register unsigned int i;
  register float save ;
  int incx, incy;

  float *x = (float *)X;
  float *y = (float *)Y;

  for (i=0;i < 2*N;i += 1)
  {
    incx = i*incX;
    incy = i*incY;

    save = y [incy] ;
    y [incy] = x [incx] ;
    x [incx] = save ;
  }
}

void mncblas_cswap_omp(const int N, void *X, const int incX,void *Y, const int incY)
{
  register unsigned int i;
  register float save ;
  int incx, incy;

  float *x = (float *)X;
  float *y = (float *)Y;

  #pragma omp for schedule(static) private(save, incx, incy)
  for (i=0;i < 2*N;i += 1)
  {
    incx = i*incX;
    incy = i*incY;

    save = y [incy] ;
    y[incy] = x [incx] ;
    x [incx] = save ;
  }
}

/*************************** ZSWAP **************************/

void mncblas_zswap_noomp(const int N, void *X, const int incX,void *Y, const int incY)
{
  register unsigned int i;
  register double save ;
  int incx, incy;

  double *x = (double *)X;
  double *y = (double *)Y;

  for (i=0;i < 2*N;i += 1)
  {
    incx = i*incX;
    incy = i*incY;

    save = y [incy] ;
    y [incy] = x [incx] ;
    x [incx] = save ;
  }
}

void mncblas_zswap_omp(const int N, void *X, const int incX,void *Y, const int incY)
{
  register unsigned int i;
  register double save ;
  int incx, incy;

  double *x = (double *)X;
  double *y = (double *)Y;

  #pragma omp for schedule(static) private(save, incx, incy)
  for (i=0;i < 2*N;i += 1)
  {
    incx = i*incX;
    incy = i*incY;

    save = y [incy] ;
    y [incy] = x [incx] ;
    x [incx] = save ;
  }
}

void mncblas_zswap_vec(const int N, void *X, const int incX,void *Y, const int incY)
{
  register unsigned int i;

  __m128d x, y;
  int incx, incy;

  double *a = (double *)X;
  double *b = (double *)Y;

  #pragma omp for schedule(static) private(incx, incy)
  for (i=0;i < 2*N;i += 2)
  {
    incx = i*incX;
    incy = i*incY;
    x = _mm_load_pd (a+incx) ;
    y = _mm_load_pd (b+incy) ;
    _mm_store_pd (b+incy, x) ;
    _mm_store_pd (a+incx, y) ;
  }
}
