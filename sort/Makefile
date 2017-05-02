OPTIONS=-fopenmp

all : mergesort qsort bubble clean

mergesort : mergesort.o
	gcc -o mergesort mergesort.o $(OPTIONS)

mergesort.o : mergesort.c
	gcc $(OPTIONS) -c mergesort.c

qsort : qsort.o
	gcc -o qsort qsort.o $(OPTIONS)

qsort.o : qsort.c
	gcc $(OPTIONS) -c qsort.c

bubble : bubble.o
	gcc -o bubble bubble.o $(OPTIONS)

bubble.o : bubble.c
	gcc $(OPTIONS) -c bubble.c

clean :
	rm -f *.o