#include <stdio.h>
#include <cblas.h>

#include "mnblas.h"

/*
  Mesure des cycles
*/

#include <x86intrin.h>

#define NBEXPERIMENTS    102
static long long unsigned int experiments [NBEXPERIMENTS] ;

// #define VECSIZE    32
#define VECSIZE    1048576

typedef float vfloat  [VECSIZE] __attribute__ ((aligned (16))) ;
typedef double vdouble [VECSIZE] __attribute__ ((aligned (16))) ;

vdouble vec1, vec2 ;

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


void vector_init (vdouble V, double x)
{
  register unsigned int i ;

  for (i = 0; i < VECSIZE; i++)
    V [i] = x ;

  return ;
}

void vector_print (vdouble V)
{
  register unsigned int i ;

  for (i = 0; i < VECSIZE; i++)
    printf ("%f ", V[i]) ;
  printf ("\n") ;

  return ;
}

int main (int argc, char **argv)
{
  unsigned long long int start, end ;
  unsigned long long int residu ;
  unsigned long long int av ;
  int exp ;
  printf("Comparaison pour DDOT entre CBLAS, notre fonction non parallélisée et notre fonction parallelisée\n");
 /* Calcul du residu de la mesure */
  start = _rdtsc () ;
  end = _rdtsc () ;
  residu = end - start ;

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      vector_init (vec1, 1.0) ;

      start = _rdtsc () ;

         cblas_ddot (VECSIZE, vec1, 1, vec2, 1) ;

      end = _rdtsc () ;

      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("cblas_ddot : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,((((double) 2 * (double) VECSIZE)) / ((double) (av - residu) * (double) 0.38)));


  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      vector_init (vec1, 1.0) ;

      start = _rdtsc () ;

         mncblas_ddot_noomp (VECSIZE, vec1, 1, vec2, 1) ;

      end = _rdtsc () ;

      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("mncblas_ddot_noomp : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,((((double) 2 * (double) VECSIZE)) / ((double) (av - residu) * (double) 0.38)));


  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      vector_init (vec1, 1.0) ;
      vector_init (vec2, 2.0) ;

      start = _rdtsc () ;

          mncblas_ddot_omp (VECSIZE, vec1, 1, vec2, 1) ;

      end = _rdtsc () ;

      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  // vector_print (vec2) ;
  printf ("mncblas_ddot_omp : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,((((double) 2 * (double) VECSIZE)) / ((double) (av - residu) * (double) 0.38)));

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      vector_init (vec1, 1.0) ;
      vector_init (vec2, 2.0) ;

      start = _rdtsc () ;

          mncblas_ddot_vec (VECSIZE, vec1, 1, vec2, 1) ;

      end = _rdtsc () ;

      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  // vector_print (vec2) ;
  printf ("mncblas_ddot_vec : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,((((double) 2 * (double) VECSIZE)) / ((double) (av - residu) * (double) 0.38)));




}
