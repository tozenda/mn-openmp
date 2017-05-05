#include "mnblas.h"

void mncblas_scopy(const int N, const float *X, const int incX, float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  for (; ((i < N) && (j < N)) ; i = i + incX + 4, j = j + incY + 4)
  {
    Y [j] = X [i] ;
    Y [j+1] = X [i+1] ;
    Y [j+2] = X [i+2] ;
    Y [j+3] = X [i+3] ;
  }
  return ;
}

void mncblas_scopy_noomp(const int N, const float *X, const int incX, float *Y, const int incY){
  register unsigned int i = 0 ;
  for (i=0;i < N; i = i + incX + 4){
    Y [i] = X [i] ;
    Y [i+1] = X [i+1] ;
    Y [i+2] = X [i+2] ;
    Y [i+3] = X [i+3] ;
  }

  return ;
}

void mncblas_scopy_omp(const int N, const float *X, const int incX, float *Y, const int incY){
  register unsigned int i = 0 ;
  #pragma omp for schedule(static)
  for (i=0;i < N; i = i + incX + 4){
    Y [i] = X [i] ;
    Y [i+1] = X [i+1] ;
    Y [i+2] = X [i+2] ;
    Y [i+3] = X [i+3] ;
  }

  return ;
}

void mncblas_dcopy(const int N, const double *X, const int incX, double *Y, const int incY){
  register unsigned int i = 0 ;

  for (i=0;i < N; i = i + incX + 4){
    Y [i] = X [i] ;
    Y [i+1] = X [i+1] ;
    Y [i+2] = X [i+2] ;
    Y [i+3] = X [i+3] ;
  }

  return ;
}

void mncblas_dcopy_noomp(const int N, const double *X, const int incX, double *Y, const int incY){
  register unsigned int i = 0 ;

  for (i=0;i < N; i = i + incX + 4){
    Y [i] = X [i] ;
    Y [i+1] = X [i+1] ;
    Y [i+2] = X [i+2] ;
    Y [i+3] = X [i+3] ;
  }

  return ;
}

void mncblas_dcopy_omp(const int N, const double *X, const int incX, double *Y, const int incY){
  register unsigned int i = 0 ;
  #pragma omp for schedule(static)
  for (i=0;i < N; i = i + incX + 4){
    Y [i] = X [i] ;
    Y [i+1] = X [i+1] ;
    Y [i+2] = X [i+2] ;
    Y [i+3] = X [i+3] ;
  }
  return ;
}

void mncblas_ccopy_noomp(const int N, const void *X, const int incX, void *Y, const int incY)
{
  float *x = (float *) X;
  float *y = (float *) Y;

  register unsigned int i;

  for(i=0;i<N;i = i + incX + 4){
    y[i] = x[i];
    y[i+1] = x[i+1];
    y[i+2] = x[i+2];
    y[i+3] = x[i+3];
  }
}

void mncblas_ccopy_omp(const int N, const void *X, const int incX, void *Y, const int incY)
{
  float *x = (float *) X;
  float *y = (float *) Y;

  register unsigned int i;
  #pragma omp parallel for schedule(static) private(i)
  for(i=0;i<N;i = i + incX + 4){
    y[i] = x[i];
    y[i+1] = x[i+1];
    y[i+2] = x[i+2];
    y[i+3] = x[i+3];
  }
}

void mncblas_ccopy_vec(const int N, const void *X, const int incX, void *Y, const int incY){
  //TODO
}

void mncblas_zcopy_noomp(const int N, const void *X, const int incX,void *Y, const int incY)
{
  double *x = (double *) X;
  double *y = (double *) Y;
  register unsigned int i;
  for(i=0;i<N;i = i + incX + 4){
    y[i] = x[i];
    y[i+1] = x[i+1];
    y[i+2] = x[i+2];
    y[i+3] = x[i+3];
  }
}

void mncblas_zcopy_omp(const int N, const void *X, const int incX,void *Y, const int incY){
  double *x = (double *) X;
  double *y = (double *) Y;
  register unsigned int i;
  #pragma omp parallel for schedule(static) private(i)
  for(i=0;i<N;i = i + incX + 4){
    y[i] = x[i];
    y[i+1] = x[i+1];
    y[i+2] = x[i+2];
    y[i+3] = x[i+3];
  }
}

void mncblas_zdot_vec(const int N, const void *X, const int incX,void *Y, const int incY){
  //TODO
}
