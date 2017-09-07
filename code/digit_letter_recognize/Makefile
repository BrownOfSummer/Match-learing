CPP=g++
CC=gcc
INCLUDE=$(shell pkg-config --cflags opencv)
LIBS=$(shell pkg-config --libs opencv)
OBJS=
BIN=pre-process select_roi_mouse
all:
	make pre-process
	make select_roi_mouse
pre-process: pre-process.o $(OBJS)
	$(CPP) -O3 -o pre-process pre-process.o $(OBJS) $(LIBS)
select_roi_mouse: select_roi_mouse.o $(OBJS)
	$(CPP) -O3 -o select_roi_mouse select_roi_mouse.o $(OBJS) $(LIBS)
.cpp.o:
	$(CPP) $(INCLUDE) -c $< -o $@
.c.o:
	$(CC) $(INCLUDE) -c $< -o $@
clean:
	rm *.o
	rm $(BIN)
