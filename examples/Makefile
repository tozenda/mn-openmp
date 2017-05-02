BINARIES=  mntest1 mntest1dyn mntest2 mntest2dyn cblastest1 cblastest2 cblastest3 mntest3 mntest3dyn

all: $(BINARIES)

INC_DIR=../include

LIB_DIR=../lib

LIBST=-lmnblas
LIBDYN=-lmnblasdyn

OPTIONS_COMPIL  =-O2 -fPIC -I$(INC_DIR)

OPTIONS_LINK_STATIC  =   -L$(LIB_DIR) $(LIBST) 
OPTIONS_LINK_DYN  = -L$(LIB_DIR) $(LIBDYN)
OPTIONS_LINK_CBLAS = -lgsl -lgslcblas

cblastest3: cblastest3.o
	gcc -o cblastest3 cblastest3.o $(OPTIONS_LINK_CBLAS)

cblastest2: cblastest2.o
	gcc -o cblastest2 cblastest2.o $(OPTIONS_LINK_CBLAS)

cblastest1: cblastest1.o
	gcc -o cblastest1 cblastest1.o $(OPTIONS_LINK_CBLAS)

mntest1: mntest1.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest1 mntest1.o $(OPTIONS_LINK_STATIC)

mntest1dyn: mntest1.o $(LIB_DIR)/libmnblasdyn.so
	gcc -o mntest1dyn mntest1.o $(OPTIONS_LINK_DYN)

mntest2dyn: mntest2.o $(LIB_DIR)/libmnblasdyn.so
	gcc -o mntest2dyn mntest2.o $(OPTIONS_LINK_DYN)

mntest2: mntest2.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest2 mntest2.o $(OPTIONS_LINK_STATIC)

mntest3dyn: mntest3.o $(LIB_DIR)/libmnblasdyn.so
	gcc -o mntest3dyn mntest2.o $(OPTIONS_LINK_DYN)

cblastest1.o: cblastest1.c
	gcc $(OPTIONS_COMPIL) -c cblastest1.c

cblastest2.o: cblastest2.c
	gcc $(OPTIONS_COMPIL) -c cblastest2.c

cblastest3.o: cblastest3.c
	gcc $(OPTIONS_COMPIL) -c cblastest3.c

mntest3: mntest3.o $(LIB_DIR)/libmnblas.a
	gcc -o mntest3 mntest3.o $(OPTIONS_LINK_STATIC)

mntest1.o: mntest1.c
	gcc $(OPTIONS_COMPIL) -c mntest1.c

mntest2.o: mntest2.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS_COMPIL) -c mntest2.c

mntest3.o: mntest3.c $(INC_DIR)/mnblas.h
	gcc $(OPTIONS_COMPIL) -c mntest3.c

clean:
	rm -f *.o $(BINARIES) *~
