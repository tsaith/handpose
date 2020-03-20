# Make macros:
# $< = first input
# $^ = inputs
# $@ = outputs

CC  = g++
CFLAGS  =
CPPFLAGS = -Iinclude/ -pthread  -O3 -Wall -fPIC -g -fwrapv -std=c++11 -m64
LDFLAGS  = -lmadgwick_ahrs
CLINKER  = $(CC)
LIBPATH  = -L/usr/local/ -L./
AR  = ar crl
RANLIB  = ranlib

all: madgwick_ahrs.o test_madgwick.o
	${CC} ${CPPFLAGS} $^ -o test

madgwick_ahrs.o: madgwick_ahrs.cpp
	${CC} ${CPPFLAGS} -c $^ -o $@

test_madgwick.o: test_madgwick.cpp
	${CC} ${CPPFLAGS} -c $^ -o $@

clean:
	rm -f *.o *.so
