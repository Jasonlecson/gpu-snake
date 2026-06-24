# GPU Snake

A snake game where the game logic runs entirely on the GPU via OpenCL. The grid size adapts to your terminal dimensions.

## Features

- Game state update (snake movement, collision detection, food spawning) runs on GPU
- Terminal-adaptive grid size (terminal width x height = grid size)
- Real-time GPU performance metrics (kernel execution time, queue delay, throughput)
- Auto-detects GPU across all OpenCL platforms (AMD, NVIDIA, Intel, etc.)
- **Cross-platform**: Linux, macOS, Windows

## Requirements

- OpenCL runtime
- ncurses (Linux/macOS) or pdcurses (Windows)
- C compiler (gcc, clang, or MSVC)

### Linux (Debian/Ubuntu)

```bash
sudo apt install build-essential libncurses-dev ocl-icd-opencl-dev opencl-headers
```

### Linux (Fedora)

```bash
sudo dnf install gcc ncurses-devel ocl-icd-devel opencl-headers
```

### Linux (Arch)

```bash
sudo pacman -S gcc ncurses ocl-icd opencl-headers
```

### macOS

```bash
# OpenCL is built-in, ncurses is pre-installed
# Just need Xcode command line tools
xcode-select --install
```

### Windows (MSYS2/MinGW)

```bash
# Install MSYS2 from https://www.msys2.org
# Then in MSYS2 MinGW64 shell:
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-opencl-icd-loader mingw-w64-x86_64-pdcurses
```

### Windows (Visual Studio)

Install the OpenCL SDK from your GPU vendor (NVIDIA/AMD/Intel), then compile with:

```cmd
cl gpu_snake.c /I"path\to\opencl\include" /link OpenCL.lib pdcurses.lib
```

## Build & Run

### Linux / macOS

```bash
make
./gpu_snake
```

If you get permission errors accessing the GPU (Linux):

```bash
sg render -c "./gpu_snake"
```

### Windows (MSYS2)

```bash
make
./gpu_snake.exe
```

### Manual compilation

```bash
# Linux
gcc -O2 -o gpu_snake gpu_snake.c -lOpenCL -lncursesw -lm

# macOS
gcc -O2 -o gpu_snake gpu_snake.c -framework OpenCL -lncurses

# Windows (MinGW)
gcc -O2 -o gpu_snake.exe gpu_snake.c -lOpenCL -lpdcurses
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

## License

MIT
