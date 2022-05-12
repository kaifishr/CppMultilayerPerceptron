CC = g++
CFLAGS = -O3 -Wall -lm

src = $(wildcard *.cpp)
OBJ = $(src:.cpp=.o)

a.out: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean
clean:
	rm -f $(OBJ) a.out ../visualization/*.png ../results/*.dat
