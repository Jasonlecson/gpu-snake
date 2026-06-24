CC = gcc
CFLAGS = -O2 -Wall
TARGET = gpu_snake

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
  LDFLAGS = -lOpenCL -lncursesw -lm
else ifeq ($(UNAME_S),Darwin)
  LDFLAGS = -framework OpenCL -lncurses -lm
else
  # Windows (MinGW/MSYS2) - uses native Console API, no ncurses needed
  LDFLAGS = -lOpenCL
endif

all: $(TARGET)

$(TARGET): gpu_snake.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET).exe

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
