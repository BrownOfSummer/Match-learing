CC=gcc
CPP=g++
INCLUDE=$(shell pkg-config --cflags opencv)
LIBS=$(shell pkg-config --libs opencv)
all:
	make generate_text_in_image
	make Drawing
	make resize_pad_to_fixed

generate_text_in_image: generate_text_in_image.o
	$(CPP) -O3 -o generate_text_in_image generate_text_in_image.o $(LIBS)
Drawing: Drawing.o
	$(CPP) -O3 -o Drawing Drawing.o $(LIBS)
resize_pad_to_fixed: resize_pad_to_fixed.o
	$(CPP) -O3 -o resize_pad_to_fixed resize_pad_to_fixed.o $(LIBS)

.cpp.o:
	$(CPP) $(INCLUDE) -c $< -o $@
clean:
	rm *.o
	rm generate_text_in_image
	rm Drawing
	rm resize_pad_to_fixed
