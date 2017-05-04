#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include<omp.h>

#include <x86intrin.h>

#define NBEXPERIMENTS   7

static long long unsigned int experiments [NBEXPERIMENTS] ;

static   unsigned int N ;

typedef  int  *array_int ;

static array_int X ;

void init_array (array_int T)
{
  register int i ;

  for (i = 0 ; i < N ; i++)
    {
      T [i] = N - i ;
    }
}

void print_array (array_int T)
{
  register int i ;

  for (i = 0 ; i < N ; i++)
    {
      printf ("%d ", T[i]) ;
    }
  printf ("\n") ;
}

int is_sorted (array_int T)
{
  register int i ;

  for (i = 1 ; i < N ; i++)
    {
      if (T[i-1] > T [i])
	return 0 ;
    }
  return 1 ;
}

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

void fusion(int *T,int deb1,int fin1,int fin2){
    int *table1;
    int deb2=fin1+1;
    int compt1=deb1;
    int compt2=deb2;
    int i;

    table1=malloc((fin1-deb1+1)*sizeof(int));

    //on recopie les éléments du début du tableau
    for(i=deb1;i<=fin1;i++){
        table1[i-deb1]=T[i];
    }

    for(i=deb1;i<=fin2;i++){
        if (compt1==deb2){
            break; //tous les éléments ont été classés
        }
        else if (compt2==(fin2+1)){ //tous les éléments du 2eme tableau ont été utilisés
            T[i]=table1[compt1-deb1]; //on ajoute les éléments restants du premier tableau
                compt1++;
        }
        else if (table1[compt1-deb1]<T[compt2]){
            T[i]=table1[compt1-deb1]; //on ajoute un élément du premier tableau
            compt1++;
        }
        else{
            T[i]=T[compt2]; //on ajoute un élément du second tableau
            compt2++;
        }
    }
    free(table1);
}

void tri_fusion_bis(int *T,int deb,int fin){
    if (deb!=fin){
        int milieu=(fin+deb)/2;
        tri_fusion_bis(T,deb,milieu);
        tri_fusion_bis(T,milieu+1,fin);
        fusion(T,deb,milieu,fin);
    }
}

void merge_sort (int *T, const int size)
{
    /* TODO: sequential version of the merge sort algorithm */
    {
    if (size>0){
        tri_fusion_bis(T,0,size-1);
        }
    }
}

void fusion1(int *T,int deb1,int fin1,int fin2){
    int *table1;
    int deb2=fin1+1;
    int compt1=deb1;
    int compt2=deb2;
    int i;

    table1=malloc((fin1-deb1+1)*sizeof(int));

    //on recopie les éléments du début du tableau
    #pragma omp parallel for schedule (static)
    for(i=deb1;i<=fin1;i++){
        table1[i-deb1]=T[i];
    }

    for(i=deb1;i<=fin2;i++){
        if (compt1==deb2){
            break; //tous les éléments ont été classés
        }
        else if (compt2==(fin2+1)){ //tous les éléments du 2eme tableau ont été utilisés
            T[i]=table1[compt1-deb1]; //on ajoute les éléments restants du premier tableau
                compt1++;
        }
        else if (table1[compt1-deb1]<T[compt2]){
            T[i]=table1[compt1-deb1]; //on ajoute un élément du premier tableau
            compt1++;
        }
        else{
            T[i]=T[compt2]; //on ajoute un élément du second tableau
            compt2++;
        }
    }
    free(table1);
}

void tri_fusion_bis1(int *T,int deb,int fin){
    if (deb!=fin){
        int milieu=(fin+deb)/2;
        tri_fusion_bis1(T,deb,milieu);
        tri_fusion_bis1(T,milieu+1,fin);
        fusion1(T,deb,milieu,fin);
    }
}


void parallel_merge_sort (int *T, const int size)
{
    /* TODO: sequential version of the merge sort algorithm */
    {
    if (size>0){
        tri_fusion_bis1(T,0,size-1);
        }
    }
}


int main (int argc, char **argv)
{
  unsigned long long int start, end, residu ;
  unsigned long long int av ;
  unsigned int exp ;

    if (argc != 2)
    {
      fprintf (stderr, "mergesort N \n") ;
      exit (-1) ;
    }

  N = 1 << (atoi(argv[1])) ;
  X = (int *) malloc (N * sizeof(int)) ;

  printf("--> Sorting an array of size %u\n",N);

  start = _rdtsc () ;
  end   = _rdtsc () ;
  residu = end - start ; 

  // print_array (X) ;

  printf("sequential sorting ...\n");


    for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      init_array (X) ;

      start = _rdtsc () ;

               merge_sort (X, N) ;

      end = _rdtsc () ;
      experiments [exp] = end - start ;

      if (! is_sorted (X))
	{
            fprintf(stderr, "ERROR: the array is not properly sorted\n") ;
            exit (-1) ;
	}
    }

  av = average (experiments) ;
  printf ("\n merge sort serial\t\t %Ld cycles\n\n", av-residu) ;

  printf("parallel sorting ...\n");

  for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
    {
      init_array (X) ;

      start = _rdtsc () ;

           parallel_merge_sort (X, N) ;

      end = _rdtsc () ;
      experiments [exp] = end - start ;

      if (! is_sorted (X))
	{
            fprintf(stderr, "ERROR: the array is not properly sorted\n") ;
            exit (-1) ;
	}
    }

  av = average (experiments) ;
  printf ("\n merge sort parallel with tasks\t %Ld cycles\n\n", av-residu) ;


}
