CC ?= cc
CFLAGS = -O3 -ffast-math -Wall -Wpedantic -Wextra -std=gnu11 -march=native -mtune=native
TARGET = calculate
BLAS_LIBS = $(shell pkg-config --cflags --libs blas)

all: $(TARGET)

$(TARGET): src/$(TARGET).c
	$(CC) $(CFLAGS) $(BLAS_LIBS) -o $(TARGET) src/$(TARGET).c
	strip $(TARGET)

clean:
	$(RM) $(TARGET)
