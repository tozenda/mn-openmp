#include <stdio.h>
#include <string.h>
#include "mnblas.h"
#include <cblas.h>

/*
  Mesure des cycles
*/

#include <x86intrin.h>
#define NBEXPERIMENTS 10
//#define NBEXPERIMENTS    252
static long long unsigned int experiments [NBEXPERIMENTS] ;

#define VECSIZE 64
//#define VECSIZE    512

typedef double mfloat  [VECSIZE] [VECSIZE] ;
typedef double mdouble [VECSIZE] [VECSIZE] ;

typedef double vfloat  [VECSIZE] ;
typedef double vdouble [VECSIZE] ;

mfloat A ;
vfloat X, Y ;

long long unsigned int average (long long unsigned int *exps)
{
  unsigned int i ;
  long long unsigned int s = 0 ;

  for (i = 2; i < (NBEXPERIMENTS-2); i++)
    {
      s = s + exps [i] ;
    }

  return s / (NBEXPERIMENTS-2) ;
}


void vector_float_init (vfloat V, float x)
{
  register unsigned int i ;

  for (i = 0; i < VECSIZE; i++)
    V [i] = x ;

  return ;
}

void vector_double_init (vdouble V, double x)
{
  register unsigned int i ;

  for (i = 0; i < VECSIZE; i++)
    V [i] = x ;

  return ;
}

void matrix_float_init (mfloat M, float x)
{
  register unsigned int i ;
  register unsigned int j ;

  for (i = 0; i < VECSIZE; i++)
      for (j = 0; j < VECSIZE; j++)
	M [i][j] = x ;

  return ;
}

void matrix_double_init (mdouble M, double x)
{
  register unsigned int i ;
  register unsigned int j ;

  for (i = 0; i < VECSIZE; i++)
      for (j = 0; j < VECSIZE; j++)
	M [i][j] = x ;

  return ;
}


void vector_float_print (vfloat V)
{
  register unsigned int i ;

  for (i = 0; i < VECSIZE; i++)
    printf ("%f ", V[i]) ;
  printf ("\n") ;

  return ;
}

void vector_double_print (vdouble V)
{
  register unsigned int i ;

  for (i = 0; i < VECSIZE; i++)
    printf ("%f ", V[i]) ;
  printf ("\n") ;

  return ;
}

void matrix_float_print (mfloat M)
{
  register unsigned int i ;
  register unsigned int j ;


  for (i = 0; i < VECSIZE; i++)
    {
      for (j = 0 ; j < VECSIZE; j++)
	printf ("%f ", M[i][j]) ;
      printf ("\n") ;
    }
  printf ("\n") ;

  return ;
}

void matrix_double_print (mdouble M)
{
  register unsigned int i ;
  register unsigned int j ;


  for (i = 0; i < VECSIZE; i++)
    {
      for (j = 0 ; j < VECSIZE; j++)
	printf ("%f ", M[i][j]) ;
      printf ("\n") ;
    }
  printf ("\n") ;

  return ;
}


int main (int argc, char **argv)
{
 unsigned long long start, end ;
 unsigned long long residu ;
 unsigned long long int av ;

 int exp ;

 /* Calcul du residu de la mesure */
  start = _rdtsc () ;
  end = _rdtsc () ;
  residu = end - start ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      vector_double_init (X, 1.0) ;
      vector_double_init (Y, 2.0) ;
      matrix_double_init (A, 3.0) ;

      start = _rdtsc () ;

         cblas_dgemv  (CblasRowMajor, CblasNoTrans, VECSIZE, VECSIZE, 1.0,(double *) A, VECSIZE, (double *) X, 1, 1.0, (double *) Y, 1) ;

      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
  vector_double_print(Y);

  printf ("cblas_dgemv : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,((((double) 4 * (double) VECSIZE) + ((double) 2 * (double) VECSIZE * (double) VECSIZE)) / ((double) (av - residu) * (double) 0.17)));


  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      vector_double_init (X, 1.0) ;
      vector_double_init (Y, 2.0) ;
      matrix_double_init (A, 3.0) ;

      start = _rdtsc () ;

      mncblas_dgemv_noomp(CblasRowMajor, CblasNoTrans, VECSIZE, VECSIZE, 1.0,
			  (double *) A, VECSIZE, (double *) X, 1, 1.0, (double *) Y, 1) ;
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
  printf ("mncblas_dgemv_noomp : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,((((double) 4 * (double) VECSIZE) + ((double) 2 * (double) VECSIZE * (double) VECSIZE)) / ((double) (av - residu) * (double) 0.17)));


  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      vector_double_init (X, 1.0) ;
      vector_double_init (Y, 2.0) ;
      matrix_double_init (A, 3.0) ;

      start = _rdtsc () ;

         mncblas_dgemv_omp(CblasRowMajor, CblasNoTrans, VECSIZE, VECSIZE, 1.0,
			  (double *) A, VECSIZE, (double *) X, 1, 1.0, (double *) Y, 1) ;
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  vector_double_print(Y);

  printf ("mncblas_dgemv_omp : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,((((double) 4 * (double) VECSIZE) + ((double) 2 * (double) VECSIZE * (double) VECSIZE)) / ((double) (av - residu) * (double) 0.17)));

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      vector_double_init (X, 1.0) ;
      vector_double_init (Y, 2.0) ;
      matrix_double_init (A, 3.0) ;

      start = _rdtsc () ;

         mncblas_dgemv_vec(CblasRowMajor, CblasNoTrans, VECSIZE, VECSIZE, 1.0,
        (double *) A, VECSIZE, (double *) X, 1, 1.0, (double *) Y, 1) ;
      end = _rdtsc () ;
      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  vector_double_print(Y);
  printf ("mncblas_dgemv_vec : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,((((double) 4 * (double) VECSIZE) + ((double) 2 * (double) VECSIZE * (double) VECSIZE)) / ((double) (av - residu) * (double) 0.17)));
}
