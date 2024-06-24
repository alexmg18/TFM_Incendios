# CC = gcc
# CFLAGS = -fopenmp
# LDFLAGS = -lm

# all: interpolate_omp

# interpolate_omp: interpolate_omp.o
# 	$(CC) $(CFLAGS) -o interpolate_omp interpolate_omp.o $(LDFLAGS)

# interpolate_omp.o: interpolate_omp.c
# 	$(CC) $(CFLAGS) -c interpolate_omp.c

# clean:
# 	rm -f *.o interpolate_omp


CC = gcc
CFLAGS = -fopenmp -O3
LDFLAGS = -lm
SRC = interpolate_omp.c
OBJ = $(SRC:.c=.o)
TARGET = interpolate_omp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	$(RM) $(OBJ) $(TARGET)