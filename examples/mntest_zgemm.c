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

typedef double mdouble  [MSIZE] [MSIZE]  __attribute__ ((aligned (16))) ;

typedef double mdouble [MSIZE] [MSIZE]  __attribute__ ((aligned (16))) ;

mdouble A, B, C ;

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

void matrix_double_init (mdouble M, double x)
{
  register unsigned int i ;
  register unsigned int j ;

  for (i = 0; i < MSIZE; i++)
      for (j = 0; j < MSIZE; j++)
  M [i][j] = x ;

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
  unsigned long long int start, end ;
  unsigned long long int residu ;
  unsigned long long int av ;
  float alpha[2];
  alpha[0] = 1.0;
  alpha[1] = 1.0;
  float beta[2];
  beta[0] = 1.0;
  beta[1] = 1.0;
  int exp ;
  printf("Comparaison pour GEMM entre CBLAS, notre fonction non parallélisée, parallélisée et notre fonction parallelisée\n");
 /* Calcul du residu de la mesure */
  start = _rdtsc () ;
  end = _rdtsc () ;
  residu = end - start ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      matrix_double_init (A, 1.0) ;
      matrix_double_init (B, 2.0) ;
      matrix_double_init (C, 3.0) ;

      start = _rdtsc () ;

         cblas_zgemm  (
       MNCblasRowMajor, MNCblasNoTrans,  MNCblasNoTrans,
       MSIZE/2, MSIZE/2, MSIZE/2,(void *) alpha, (float *) A, MSIZE/2,
       (float *) B, MSIZE/2,(void *) beta, (float *) C, MSIZE/2
            ) ;

      end = _rdtsc () ;

      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("cblas_zgemm : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,(((double) MSIZE * (double) MSIZE * (double) MSIZE) / ((double) (av - residu) * (double) 0.17)));

/*
  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      matrix_double_init (A, 1.0) ;
      matrix_double_init (B, 2.0) ;
      matrix_double_init (C, 3.0) ;

      start = _rdtsc () ;

         mncblas_zgemm_noomp  (
       MNCblasRowMajor, MNCblasNoTrans,  MNCblasNoTrans,
       MSIZE, MSIZE, MSIZE,(void *) alpha, (float *) A, MSIZE,
       (float *) B, MSIZE,(void *) beta, (float *) C, MSIZE
            ) ;

      end = _rdtsc () ;

      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("mncblas_zgemm_noomp : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,(((double) MSIZE * (double) MSIZE * (double) MSIZE) / ((double) (av - residu) * (double) 0.17)));


  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      matrix_double_init (A, 1.0) ;
      matrix_double_init (B, 2.0) ;
      matrix_double_init (C, 3.0) ;

      start = _rdtsc () ;

          mncblas_zgemm_omp  (
        MNCblasRowMajor, MNCblasNoTrans,  MNCblasNoTrans,
        MSIZE, MSIZE, MSIZE,(void *) alpha, (float *) A, MSIZE,
        (float *) B, MSIZE,(void *) beta, (float *) C, MSIZE
             ) ;

      end = _rdtsc () ;

      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  // vector_print (vec2) ;
  printf ("mncblas_zgemm_omp : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,(((double) MSIZE * (double) MSIZE * (double) MSIZE) / ((double) (av - residu) * (double) 0.17)));

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      matrix_double_init (A, 1.0) ;
      matrix_double_init (B, 2.0) ;
      matrix_double_init (C, 3.0) ;

      start = _rdtsc () ;

          mncblas_zgemm_vec  (
        MNCblasRowMajor, MNCblasNoTrans,  MNCblasNoTrans,
        MSIZE, MSIZE, MSIZE,(void *) alpha, (float *) A, MSIZE,
        (float *) B, MSIZE,(void *) beta, (float *) C, MSIZE
             ) ;

      end = _rdtsc () ;

      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  // vector_print (vec2) ;
  printf ("mncblas_zgemm_vec (vectorisé) : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,(((double) MSIZE * (double) MSIZE * (double) MSIZE) / ((double) (av - residu) * (double) 0.17)));
*/
}
