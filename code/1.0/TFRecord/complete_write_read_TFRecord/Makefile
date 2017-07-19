CC=g++
INCLUDE=$(shell pkg-config --cflags opencv)
LIBS=$(shell pkg-config --libs opencv)
BINS=pre_process
OBJS= 
all:
	make pre_process
pre_process: pre_process.o $(OBJS)
	$(CC) -O3 -o pre_process pre_process.o $(OBJS) $(LIBS)
.cpp.o:
	$(CC) $(INCLUDE) -c $< -o $@
.c.o:
	$(CC) $(INCLUDE) -c $< -o $@
clean:
	rm *.o
	rm $(BINS)
