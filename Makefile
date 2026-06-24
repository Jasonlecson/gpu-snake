CC = gcc
CFLAGS = -O2 -Wall
LDFLAGS = -lOpenCL -lncursesw -lm

TARGET = gpu_snake

all: $(TARGET)

$(TARGET): gpu_snake.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
