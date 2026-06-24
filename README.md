# GPU Snake

A snake game where the game logic runs entirely on the GPU via OpenCL. The grid size adapts to your terminal dimensions.

## Features

- Game state update (snake movement, collision detection, food spawning) runs on GPU
- Terminal-adaptive grid size (terminal width x height = grid size)
- Real-time GPU performance metrics (kernel execution time, queue delay, throughput)
- Auto-detects GPU across all OpenCL platforms (AMD, NVIDIA, Intel, etc.)
- Written in C with OpenCL and ncurses

## Requirements

- OpenCL runtime (Mesa Clover, NVIDIA driver, Intel NEO, pocl, etc.)
- ncurses (wide character version)
- C compiler (gcc)

### Debian/Ubuntu

```bash
sudo apt install build-essential libncurses-dev ocl-icd-opencl-dev opencl-headers
```

### Fedora

```bash
sudo dnf install gcc ncurses-devel ocl-icd-devel opencl-headers
```

### Arch

```bash
sudo pacman -S gcc ncurses ocl-icd opencl-headers
```

## Build & Run

```bash
make
./gpu_snake
```

Or directly:

```bash
gcc -O2 -o gpu_snake gpu_snake.c -lOpenCL -lncursesw -lm
./gpu_snake
```

If you get permission errors accessing the GPU:

```bash
sg render -c "./gpu_snake"
```

## Controls

- Arrow keys: move
- `q`: quit
- `r`: restart

## How It Works

Each frame, the GPU runs a kernel that processes every cell in the grid in parallel:

1. **Aging**: All snake body cells increment their age, expired cells become empty
2. **Movement**: Thread 0 calculates the new head position
3. **Collision**: Checks wall and self-collision
4. **Food**: If food is eaten, randomly spawns a new one

The kernel uses `barrier(CLK_GLOBAL_MEM_FENCE)` to synchronize between the aging and movement phases.

## GPU Auto-Detection

The game searches for OpenCL devices in this order:
1. GPU devices on all platforms (AMD, NVIDIA, Intel, etc.)
2. Any OpenCL device as fallback (e.g., pocl for CPU)

No configuration needed - it works on any system with an OpenCL runtime.
