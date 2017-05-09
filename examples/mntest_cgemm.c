#include <stdio.h>
#include <string.h>

#include "mnblas.h"
#include <cblas.h>

/*
Mesure des cycles
*/

#include <x86intrin.h>

#define NBEXPERIMENTS    52
static long long unsigned int experiments [NBEXPERIMENTS] ;

#define MSIZE    256

typedef float mfloat  [MSIZE] [MSIZE]  __attribute__ ((aligned (16))) ;

typedef double mdouble [MSIZE] [MSIZE]  __attribute__ ((aligned (16))) ;

mfloat A, B, C ;

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

void matrix_float_init (mfloat M, float x)
{
  register unsigned int i ;
  register unsigned int j ;

  for (i = 0; i < MSIZE; i++)
  for (j = 0; j < MSIZE; j++)
  if (i > j)
  M [i][j] = x ;
  else
  M [i][j] = -x ;

  return ;
}

void matrix_double_init (mdouble M, double x)
{
  register unsigned int i ;
  register unsigned int j ;

  for (i = 0; i < MSIZE; i++)
  for (j = 0; j < MSIZE; j++)
  M [i][j] = x ;

  return ;
}

void matrix_float_print (mfloat M)
{
  register unsigned int i ;
  register unsigned int j ;


  for (i = 0; i < MSIZE; i++)
  {
    for (j = 0 ; j < MSIZE; j++)
    printf ("%3.2f ", M[i][j]) ;
    printf ("\n") ;
  }
  printf ("\n") ;

  return ;
}

void matrix_double_print (mdouble M)
{
  register unsigned int i ;
  register unsigned int j ;


  for (i = 0; i < MSIZE; i++)
  {
    for (j = 0 ; j < MSIZE; j++)
    printf ("%3.2f ", M[i][j]) ;
    printf ("\n") ;
  }
  printf ("\n") ;

  return ;
}

int main (int argc, char **argv)
{
  float alpha[2];
  alpha[0] = 1.0;
  alpha[1] = 1.0;
  float beta[2];
  beta[0] = 1.0;
  beta[1] = 1.0;
  unsigned long long int start, end ;
  unsigned long long int residu ;
  unsigned long long int av ;
  int exp ;
  printf("Comparaison pour GEMM entre CBLAS, notre fonction non parallélisée et notre fonction parallelisée\n");
  /* Calcul du residu de la mesure */
  start = _rdtsc () ;
  end = _rdtsc () ;
  residu = end - start ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  {
    matrix_float_init (A, 1.0) ;
    matrix_float_init (B, 2.0) ;
    matrix_float_init (C, 3.0) ;

    start = _rdtsc () ;

    cblas_cgemm  (
      MNCblasRowMajor, MNCblasNoTrans,  MNCblasNoTrans,
      MSIZE/2, MSIZE/2, MSIZE/2,(void *) alpha, (float *) A, MSIZE/2,
      (float *) B, MSIZE/2,(void *) beta, (float *) C, MSIZE/2
    ) ;

    end = _rdtsc () ;

    experiments [exp] = end - start ;
  }

  av = average (experiments) ;

  printf ("cblas_cgemm : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,(((double) MSIZE * (double) MSIZE * (double) MSIZE) / ((double) (av - residu) * (double) 0.17)));

/*
  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  {
    matrix_float_init (A, 1.0) ;
    matrix_float_init (B, 2.0) ;
    matrix_float_init (C, 3.0) ;

    start = _rdtsc () ;

    mncblas_cgemm_noomp  (
      MNCblasRowMajor, MNCblasNoTrans,  MNCblasNoTrans,
      MSIZE, MSIZE, MSIZE,(void *) alpha, (float *) A, MSIZE,
      (float *) B, MSIZE,(void *) beta, (float *) C, MSIZE
    ) ;

    end = _rdtsc () ;

    experiments [exp] = end - start ;
  }

  av = average (experiments) ;

  printf ("mncblas_cgemm_noomp : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,(((double) MSIZE * (double) MSIZE * (double) MSIZE) / ((double) (av - residu) * (double) 0.17)));


  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  {
    matrix_float_init (A, 1.0) ;
    matrix_float_init (B, 2.0) ;
    matrix_float_init (C, 3.0) ;

    start = _rdtsc () ;

    mncblas_cgemm_omp  (

      MNCblasRowMajor, MNCblasNoTrans,  MNCblasNoTrans,
      MSIZE, MSIZE, MSIZE,(void *) alpha, (float *) A, MSIZE,
      (float *) B, MSIZE,(void *) beta, (float *) C, MSIZE
    ) ;

    end = _rdtsc () ;

    experiments [exp] = end - start ;
  }

  av = average (experiments) ;

  // vector_print (vec2) ;
  printf ("mncblas_cgemm_omp : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,(((double) MSIZE * (double) MSIZE * (double) MSIZE) / ((double) (av - residu) * (double) 0.17)));

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
  {
    matrix_float_init (A, 1.0) ;
    matrix_float_init (B, 2.0) ;
    matrix_float_init (C, 3.0) ;

    start = _rdtsc () ;

    mncblas_cgemm_vec  (
      MNCblasRowMajor, MNCblasNoTrans,  MNCblasNoTrans,
      MSIZE, MSIZE, MSIZE,(void *) alpha, (float *) A, MSIZE,
      (float *) B, MSIZE,(void *) beta, (float *) C, MSIZE
    ) ;

    end = _rdtsc () ;

    experiments [exp] = end - start ;
  }

  av = average (experiments) ;

  // vector_print (vec2) ;
  printf ("mncblas_cgemm_vec (vectorisé) : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,(((double) MSIZE * (double) MSIZE * (double) MSIZE) / ((double) (av - residu) * (double) 0.17)));
*/
}
