CPP=g++
CC=gcc
INCLUDE=$(shell pkg-config --cflags opencv)
LIBS=$(shell pkg-config --libs opencv)
OBJS=
BIN=pre-process
all:
	make pre-process
pre-process: pre-process.o $(OBJS)
	$(CPP) -O3 -o pre-process pre-process.o $(OBJS) $(LIBS)
.cpp.o:
	$(CPP) $(INCLUDE) -c $< -o $@
.c.o:
	$(CC) $(INCLUDE) -c $< -o $@
clean:
	rm *.o
	rm $(BIN)
