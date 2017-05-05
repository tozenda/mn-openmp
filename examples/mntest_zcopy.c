#include <stdio.h>
#include <cblas.h>

#include "mnblas.h"

/*
  Mesure des cycles
*/
#include <x86intrin.h>

#define NBEXPERIMENTS    102
static long long unsigned int experiments [NBEXPERIMENTS] ;

 //#define VECSIZE    64
 //#define VECSIZE    1000
#define VECSIZE    1048576

typedef double vfloat  [VECSIZE] __attribute__ ((aligned (16))) ;
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
  printf("Comparaison pour COPY entre notre fonction non parallélisée, notre fonction parallelisée et notre fonction vectorisée et parallélisée\n");
 /* Calcul du residu de la mesure */
  start = _rdtsc () ;
  end = _rdtsc () ;
  residu = end - start ;

  /*for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      vector_init (vec1, 1.0) ;
//      vector_print(vec1);

      start = _rdtsc () ;

         cblas_ccopy (VECSIZE, vec1, 1, vec2, 1) ;

      end = _rdtsc () ;

      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("cblas_ccopy : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,((((double) VECSIZE)) / ((double) (av - residu) * (double) 0.17)));
*/

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      vector_init (vec1, 1.0) ;

      start = _rdtsc () ;

         mncblas_zcopy_noomp (VECSIZE, vec1, 1, vec2, 1) ;

      end = _rdtsc () ;

      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  printf ("mncblas_zcopy_noomp : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,((((double) VECSIZE)) / ((double) (av - residu) * (double) 0.17)));


  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      vector_init (vec1, 1.0) ;
      vector_init (vec2, 2.0) ;

      start = _rdtsc () ;

          mncblas_zcopy_omp (VECSIZE, vec1, 1, vec2, 1) ;

      end = _rdtsc () ;

      experiments [exp] = end - start ;
    }

  av = average (experiments) ;

  // vector_print (vec2) ;
  printf ("mncblas_zcopy_omp : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,((((double) VECSIZE)) / ((double) (av - residu) * (double) 0.17)));

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      vector_init (vec1, 1.0) ;
      vector_init (vec2, 2.0) ;

      start = _rdtsc () ;

          mncblas_zcopy_vec (VECSIZE, vec1, 1, vec2, 1) ;

      end = _rdtsc () ;

      experiments [exp] = end - start ;
    }

  av = average (experiments) ;
  printf ("mncblas_zcopy_vec : nombre de cycles: \t %Ld ;\t GFLOP/s :\t %3.3f\n ", av-residu,((((double) VECSIZE)) / ((double) (av - residu) * (double) 0.17)));

}
