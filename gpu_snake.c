#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <locale.h>

#ifdef _WIN32
  #include <windows.h>
  #include <conio.h>
  #define USE_WINCONSOLE 1
#else
  #include <unistd.h>
  #include <sys/ioctl.h>
  #include <curses.h>
#endif

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

#define FOOD_COUNT 10

static const char *KERNEL_SRC =
"__kernel void snake_update(\n"
"    __global int *grid, __global int *head, __global int *dir,\n"
"    __global int *len, __global int *status, __global int *rng,\n"
"    int W, int H)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    int total = W * H;\n"
"    if (id >= total) return;\n"
"    int frame = len[0] + 1;\n"
"    if (grid[id] > 0) { grid[id]++; if (grid[id] > frame) grid[id] = 0; }\n"
"    barrier(CLK_GLOBAL_MEM_FENCE);\n"
"    if (id == 0) {\n"
"        int hy = head[0], hx = head[1];\n"
"        int d = dir[0];\n"
"        int ny = hy + (d == 1 ? 1 : (d == 0 ? -1 : 0));\n"
"        int nx = hx + (d == 3 ? 1 : (d == 2 ? -1 : 0));\n"
"        status[0] = 0;\n"
"        if (ny < 0 || ny >= H || nx < 0 || nx >= W) { status[0] = 2; return; }\n"
"        int target = ny * W + nx;\n"
"        int val = grid[target];\n"
"        if (val > 0) { status[0] = 2; return; }\n"
"        if (val == -1) { status[0] = 1; len[0]++; }\n"
"        grid[target] = 1;\n"
"        head[0] = ny; head[1] = nx;\n"
"    }\n"
"    barrier(CLK_GLOBAL_MEM_FENCE);\n"
"    if (id == 0 && status[0] == 1) {\n"
"        unsigned int seed = rng[0];\n"
"        int ec = 0;\n"
"        for (int i = 0; i < total; i++) if (grid[i] == 0) ec++;\n"
"        if (ec > 0) {\n"
"            seed = seed * 1103515245u + 12345u;\n"
"            int pick = seed % ec;\n"
"            int cnt = 0;\n"
"            for (int i = 0; i < total; i++) {\n"
"                if (grid[i] == 0) { if (cnt == pick) { grid[i] = -1; break; } cnt++; }\n"
"            }\n"
"            rng[0] = (int)seed;\n"
"        }\n"
"    }\n"
"}\n";

/* ======================== Platform abstraction ======================== */

static double now_us(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER cnt;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / freq.QuadPart * 1e6;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
#endif
}

static void get_terminal_size(int *w, int *h) {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info);
    *w = info.srWindow.Right - info.srWindow.Left + 1;
    *h = info.srWindow.Bottom - info.srWindow.Top + 1;
#else
    struct winsize ws;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws);
    *w = ws.ws_col;
    *h = ws.ws_row;
#endif
    if (*w < 20) *w = 20;
    if (*h < 10) *h = 10;
}

/* ======================== Windows Console API ======================== */

#ifdef USE_WINCONSOLE

static HANDLE hConsole;
static HANDLE hConsoleIn;
static COORD screenSize;
static CHAR_INFO *screenBuf;
static int console_w, console_h;

static WORD attr_map(int color_pair, int bold) {
    WORD a = 0;
    switch (color_pair) {
        case 1: a = FOREGROUND_GREEN; break;
        case 2: a = FOREGROUND_RED; break;
        case 3: a = FOREGROUND_RED | FOREGROUND_GREEN; break;
        case 4: a = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE; break;
        case 5: a = FOREGROUND_GREEN | FOREGROUND_BLUE; break;
        case 6: a = FOREGROUND_RED | FOREGROUND_BLUE; break;
        default: a = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE; break;
    }
    if (bold) a |= FOREGROUND_INTENSITY;
    return a;
}

static void console_init(void) {
    hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    hConsoleIn = GetStdHandle(STD_INPUT_HANDLE);
    SetConsoleOutputCP(65001);
    CONSOLE_CURSOR_INFO ci = {1, FALSE};
    SetConsoleCursorInfo(hConsole, &ci);
    get_terminal_size(&console_w, &console_h);
    screenBuf = malloc(console_w * console_h * sizeof(CHAR_INFO));
}

static void console_cleanup(void) {
    free(screenBuf);
    CONSOLE_CURSOR_INFO ci = {1, TRUE};
    SetConsoleCursorInfo(hConsole, &ci);
    system("cls");
}

static void console_clear(void) {
    get_terminal_size(&console_w, &console_h);
    screenBuf = realloc(screenBuf, console_w * console_h * sizeof(CHAR_INFO));
    for (int i = 0; i < console_w * console_h; i++) {
        screenBuf[i].Char.UnicodeChar = ' ';
        screenBuf[i].Attributes = 0;
    }
}

static void console_putchar(int y, int x, char ch, int color_pair, int bold) {
    if (y < 0 || y >= console_h || x < 0 || x >= console_w) return;
    int idx = y * console_w + x;
    screenBuf[idx].Char.UnicodeChar = ch;
    screenBuf[idx].Attributes = attr_map(color_pair, bold);
}

static void console_printstr(int y, int x, const char *s, int color_pair, int bold) {
    WORD a = attr_map(color_pair, bold);
    for (int i = 0; s[i] && x + i < console_w; i++) {
        if (y < 0 || y >= console_h) continue;
        int idx = y * console_w + x + i;
        screenBuf[idx].Char.UnicodeChar = s[i];
        screenBuf[idx].Attributes = a;
    }
}

static void console_printf(int y, int x, int color_pair, int bold, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    console_printstr(y, x, buf, color_pair, bold);
}

static void console_flush(void) {
    COORD bufSize = {console_w, console_h};
    COORD bufCoord = {0, 0};
    SMALL_RECT writeRegion = {0, 0, console_w - 1, console_h - 1};
    WriteConsoleOutputW(hConsole, screenBuf, bufSize, bufCoord, &writeRegion);
}

static int console_getch(void) {
    INPUT_RECORD ir;
    DWORD n;
    while (1) {
        if (!ReadConsoleInput(hConsoleIn, &ir, 1, &n)) continue;
        if (ir.EventType != KEY_EVENT) continue;
        if (!ir.Event.KeyEvent.bKeyDown) continue;
        int vk = ir.Event.KeyEvent.wVirtualKeyCode;
        switch (vk) {
            case VK_UP:    return 256 + 1;
            case VK_DOWN:  return 256 + 2;
            case VK_LEFT:  return 256 + 3;
            case VK_RIGHT: return 256 + 4;
            default:
                return ir.Event.KeyEvent.uChar.AsciiChar;
        }
    }
}

static int console_kbhit(void) {
    INPUT_RECORD ir;
    DWORD n;
    while (PeekConsoleInput(hConsoleIn, &ir, 1, &n) && n > 0) {
        if (ir.EventType == KEY_EVENT && ir.Event.KeyEvent.bKeyDown)
            return 1;
        ReadConsoleInput(hConsoleIn, &ir, 1, &n);
    }
    return 0;
}

#endif /* USE_WINCONSOLE */

/* ======================== GPU context ======================== */

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_command_queue queue;
    cl_program prog;
    cl_kernel kernel;
    cl_mem grid_g, head_g, dir_g, len_g, status_g, rng_g;
    char device_name[256];
    char platform_name[256];
    int compute_units;
    int clock_freq;
    int is_gpu;
} gpu_ctx_t;

static void check_cl(cl_int err, const char *msg) {
    if (err != CL_SUCCESS) {
#ifndef USE_WINCONSOLE
        endwin();
#endif
        fprintf(stderr, "OpenCL error %d: %s\n", err, msg);
        exit(1);
    }
}

static gpu_ctx_t gpu_init(void) {
    gpu_ctx_t g = {0};
    cl_int err;
    cl_uint num_platforms;

    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
#ifndef USE_WINCONSOLE
        endwin();
#endif
        fprintf(stderr, "No OpenCL platforms found.\n");
        exit(1);
    }

    cl_platform_id *platforms = malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);

    g.device = NULL;
    for (cl_uint p = 0; p < num_platforms && !g.device; p++) {
        cl_uint nd;
        if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &nd) == CL_SUCCESS && nd > 0) {
            cl_device_id *devs = malloc(nd * sizeof(cl_device_id));
            clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, nd, devs, NULL);
            g.device = devs[0]; g.platform = platforms[p]; g.is_gpu = 1;
            free(devs);
        }
    }
    if (!g.device) {
        for (cl_uint p = 0; p < num_platforms && !g.device; p++) {
            cl_uint nd;
            if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &nd) == CL_SUCCESS && nd > 0) {
                cl_device_id *devs = malloc(nd * sizeof(cl_device_id));
                clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, nd, devs, NULL);
                g.device = devs[0]; g.platform = platforms[p]; g.is_gpu = 0;
                free(devs);
            }
        }
    }
    free(platforms);

    if (!g.device) {
#ifndef USE_WINCONSOLE
        endwin();
#endif
        fprintf(stderr, "No OpenCL devices found\n");
        exit(1);
    }

    clGetPlatformInfo(g.platform, CL_PLATFORM_NAME, sizeof(g.platform_name), g.platform_name, NULL);
    clGetDeviceInfo(g.device, CL_DEVICE_NAME, sizeof(g.device_name), g.device_name, NULL);
    clGetDeviceInfo(g.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &g.compute_units, NULL);
    clGetDeviceInfo(g.device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(int), &g.clock_freq, NULL);

    g.ctx = clCreateContext(NULL, 1, &g.device, NULL, NULL, &err);
    check_cl(err, "create context");
#ifdef CL_VERSION_2_0
    g.queue = clCreateCommandQueueWithProperties(g.ctx, g.device,
                (cl_queue_properties[]){CL_QUEUE_PROFILING_ENABLE, 0}, &err);
    if (err != CL_SUCCESS)
#endif
    g.queue = clCreateCommandQueue(g.ctx, g.device, CL_QUEUE_PROFILING_ENABLE, &err);
    check_cl(err, "create queue");

    g.prog = clCreateProgramWithSource(g.ctx, 1, &KERNEL_SRC, NULL, &err);
    check_cl(err, "create program");
    err = clBuildProgram(g.prog, 1, &g.device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz;
        clGetProgramBuildInfo(g.prog, g.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char *log = malloc(log_sz);
        clGetProgramBuildInfo(g.prog, g.device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
#ifndef USE_WINCONSOLE
        endwin();
#endif
        fprintf(stderr, "OpenCL build error:\n%s\n", log);
        free(log);
        exit(1);
    }

    g.kernel = clCreateKernel(g.prog, "snake_update", &err);
    check_cl(err, "create kernel");
    return g;
}

static void gpu_alloc(gpu_ctx_t *g, int w, int h, int *grid) {
    cl_int err;
    g->grid_g   = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 w * h * sizeof(int), grid, &err);
    g->head_g   = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE, 2 * sizeof(int), NULL, &err);
    g->dir_g    = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
    g->len_g    = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
    g->status_g = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
    g->rng_g    = clCreateBuffer(g->ctx, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
}

static void gpu_upload(gpu_ctx_t *g, int *grid, int *head, int *dir, int *len,
                       int *status, int *rng, int w, int h) {
    clEnqueueWriteBuffer(g->queue, g->grid_g, CL_TRUE, 0, w*h*sizeof(int), grid, 0, NULL, NULL);
    clEnqueueWriteBuffer(g->queue, g->head_g, CL_TRUE, 0, 2*sizeof(int), head, 0, NULL, NULL);
    clEnqueueWriteBuffer(g->queue, g->dir_g, CL_TRUE, 0, sizeof(int), dir, 0, NULL, NULL);
    clEnqueueWriteBuffer(g->queue, g->len_g, CL_TRUE, 0, sizeof(int), len, 0, NULL, NULL);
    clEnqueueWriteBuffer(g->queue, g->status_g, CL_TRUE, 0, sizeof(int), status, 0, NULL, NULL);
    clEnqueueWriteBuffer(g->queue, g->rng_g, CL_TRUE, 0, sizeof(int), rng, 0, NULL, NULL);
}

static void gpu_free(gpu_ctx_t *g) {
    clReleaseMemObject(g->grid_g); clReleaseMemObject(g->head_g);
    clReleaseMemObject(g->dir_g);  clReleaseMemObject(g->len_g);
    clReleaseMemObject(g->status_g); clReleaseMemObject(g->rng_g);
    clReleaseKernel(g->kernel);
    clReleaseProgram(g->prog);
    clReleaseCommandQueue(g->queue);
    clReleaseContext(g->ctx);
}

/* ======================== Layout ======================== */

typedef struct {
    int grid_w, grid_h;
    int map_h;
    int info_y;
} layout_t;

static void calc_layout(layout_t *l) {
    int sw, sh;
    get_terminal_size(&sw, &sh);
    int info = sh / 4;
    if (info < 8) info = 8;
    l->info_y = sh - info;
    l->map_h = l->info_y - 1;
    l->grid_w = sw - 2;
    l->grid_h = l->map_h - 1;
    if (l->grid_w < 5) l->grid_w = 5;
    if (l->grid_h < 3) l->grid_h = 3;
}

/* ======================== Game ======================== */

static void init_game(int *grid, int *head, int *dir, int *len, int *status,
                      int *rng, int w, int h) {
    memset(grid, 0, w * h * sizeof(int));
    int sy = h / 2, sx = w / 2;
    for (int i = 0; i < 4 && sx - i >= 0; i++)
        grid[sy * w + (sx - i)] = i + 1;
    head[0] = sy; head[1] = sx;
    dir[0] = 3; len[0] = 4; status[0] = 0;
    rng[0] = (int)(time(NULL) * 1000) & 0x7FFFFFFF;
    for (int f = 0; f < FOOD_COUNT; f++) {
        int fy, fx, tries = 0;
        do { fy = rand() % h; fx = rand() % w; tries++; }
        while (grid[fy * w + fx] != 0 && tries < 1000);
        if (grid[fy * w + fx] == 0) grid[fy * w + fx] = -1;
    }
}

/* ======================== Main ======================== */

int main(void) {
    srand((unsigned)time(NULL));
    setlocale(LC_ALL, "");

    gpu_ctx_t gpu = gpu_init();
    layout_t lay;
    calc_layout(&lay);

    int gw = lay.grid_w, gh = lay.grid_h;
    int *grid = calloc(gw * gh, sizeof(int));
    int head[2], dir[1], len[1], status[1], rng[1];
    init_game(grid, head, dir, len, status, rng, gw, gh);
    gpu_alloc(&gpu, gw, gh, grid);
    gpu_upload(&gpu, grid, head, dir, len, status, rng, gw, gh);

#ifdef USE_WINCONSOLE
    console_init();
    int KEY_UP_CODE = 256 + 1;
    int KEY_DOWN_CODE = 256 + 2;
    int KEY_LEFT_CODE = 256 + 3;
    int KEY_RIGHT_CODE = 256 + 4;
#else
    initscr(); cbreak(); noecho(); curs_set(0);
    nodelay(stdscr, TRUE); timeout(80); keypad(stdscr, TRUE);
    start_color(); use_default_colors();
    init_pair(1, COLOR_GREEN, -1);
    init_pair(2, COLOR_RED, -1);
    init_pair(3, COLOR_YELLOW, -1);
    init_pair(4, COLOR_WHITE, -1);
    init_pair(5, COLOR_CYAN, -1);
    init_pair(6, COLOR_MAGENTA, -1);
#endif

    double t_sum = 0;
    int fc = 0;
    double sess = now_us();

    while (1) {
        double tf = now_us();

#ifdef USE_WINCONSOLE
        int ch = console_kbhit() ? console_getch() : -1;
        if (ch == KEY_UP_CODE)    { if (dir[0] != 1) dir[0] = 0; }
        else if (ch == KEY_DOWN_CODE)  { if (dir[0] != 0) dir[0] = 1; }
        else if (ch == KEY_LEFT_CODE)  { if (dir[0] != 3) dir[0] = 2; }
        else if (ch == KEY_RIGHT_CODE) { if (dir[0] != 2) dir[0] = 3; }
        else if (ch == 'q' || ch == 'Q') goto quit;
        else if (ch == 'r' || ch == 'R') goto restart;
        Sleep(80);
#else
        int ch = getch();
        switch (ch) {
            case KEY_UP:    if (dir[0] != 1) dir[0] = 0; break;
            case KEY_DOWN:  if (dir[0] != 0) dir[0] = 1; break;
            case KEY_LEFT:  if (dir[0] != 3) dir[0] = 2; break;
            case KEY_RIGHT: if (dir[0] != 2) dir[0] = 3; break;
            case 'q': goto quit;
            case 'r': goto restart;
        }
#endif

        layout_t nl;
        calc_layout(&nl);
        if (nl.grid_w != gw || nl.grid_h != gh) {
            int old_w = gw, old_h = gh;
            gw = nl.grid_w; gh = nl.grid_h; lay = nl;
            int *ng = calloc(gw * gh, sizeof(int));
            for (int y = 0; y < (gh < old_h ? gh : old_h); y++)
                for (int x = 0; x < (gw < old_w ? gw : old_w); x++)
                    ng[y * gw + x] = grid[y * old_w + x];
            free(grid); grid = ng;
            if (head[0] >= gh) head[0] = gh - 1;
            if (head[1] >= gw) head[1] = gw - 1;
            clReleaseMemObject(gpu.grid_g);
            cl_int err;
            gpu.grid_g = clCreateBuffer(gpu.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        gw * gh * sizeof(int), grid, &err);
            fc = 0; t_sum = 0; sess = now_us();
        }

        clEnqueueWriteBuffer(gpu.queue, gpu.dir_g, CL_TRUE, 0, sizeof(int), dir, 0, NULL, NULL);

        double t0 = now_us();
        size_t gws = (size_t)gw * gh;
        cl_event ev;
        clSetKernelArg(gpu.kernel, 0, sizeof(cl_mem), &gpu.grid_g);
        clSetKernelArg(gpu.kernel, 1, sizeof(cl_mem), &gpu.head_g);
        clSetKernelArg(gpu.kernel, 2, sizeof(cl_mem), &gpu.dir_g);
        clSetKernelArg(gpu.kernel, 3, sizeof(cl_mem), &gpu.len_g);
        clSetKernelArg(gpu.kernel, 4, sizeof(cl_mem), &gpu.status_g);
        clSetKernelArg(gpu.kernel, 5, sizeof(cl_mem), &gpu.rng_g);
        clSetKernelArg(gpu.kernel, 6, sizeof(int), &gw);
        clSetKernelArg(gpu.kernel, 7, sizeof(int), &gh);
        clEnqueueNDRangeKernel(gpu.queue, gpu.kernel, 1, NULL, &gws, NULL, 0, NULL, &ev);
        clFinish(gpu.queue);
        double t_kernel = now_us() - t0;

        t0 = now_us();
        clEnqueueReadBuffer(gpu.queue, gpu.head_g, CL_TRUE, 0, 2*sizeof(int), head, 0, NULL, NULL);
        clEnqueueReadBuffer(gpu.queue, gpu.status_g, CL_TRUE, 0, sizeof(int), status, 0, NULL, NULL);
        clEnqueueReadBuffer(gpu.queue, gpu.len_g, CL_TRUE, 0, sizeof(int), len, 0, NULL, NULL);
        clEnqueueReadBuffer(gpu.queue, gpu.grid_g, CL_TRUE, 0, gw*gh*sizeof(int), grid, 0, NULL, NULL);
        double t_rb = now_us() - t0;

        cl_ulong ps, pt, pe;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_SUBMIT, sizeof(ps), &ps, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(pt), &pt, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(pe), &pe, NULL);
        double gpu_q = (pt - ps) / 1e3;
        double gpu_k = (pe - pt) / 1e3;
        clReleaseEvent(ev);

        fc++;
        t_sum += t_kernel;

        double elapsed = (now_us() - sess) / 1e6;
        double fps = fc / elapsed;
        double avg = t_sum / fc;
        double frame_us = now_us() - tf;

#ifdef USE_WINCONSOLE
        console_clear();
        console_printf(0, 0, 6, 1, " GPU SNAKE %dx%d=%d cells | %s %s %dCU@%dMHz ",
                       gw, gh, gw * gh, gpu.platform_name, gpu.device_name,
                       gpu.compute_units, gpu.clock_freq);

        for (int y = 0; y < gh; y++) {
            for (int x = 0; x < gw; x++) {
                int v = grid[y * gw + x];
                char c; int col, bld = 0;
                if (y == head[0] && x == head[1]) { c = '@'; col = 3; bld = 1; }
                else if (v == -1) { c = '*'; col = 2; bld = 1; }
                else if (v > 0) { c = 'o'; col = 1; }
                else { c = ' '; col = 0; }
                console_putchar(y + 1, x + 1, c, col, bld);
            }
        }

        int iy = lay.info_y;
        console_printf(iy++, 0, 4, 1, " Score:%d  Length:%d  Frame:%d  Pos(%d,%d)  FPS:%.1f",
                       len[0]-4, len[0], fc, head[1], head[0], fps);
        iy++;
        console_printf(iy++, 0, 5, 0, " -- GPU -----------------------------------------------------");
        console_printf(iy++, 0, 5, 0, "  Platform:   %s", gpu.platform_name);
        console_printf(iy++, 0, 5, 0, "  Device:     %s  (%s)", gpu.device_name, gpu.is_gpu ? "GPU" : "CPU");
        console_printf(iy++, 0, 5, 0, "  CUs: %d  Freq: %d MHz  Grid: %dx%d  Threads: %d",
                       gpu.compute_units, gpu.clock_freq, gw, gh, gw * gh);
        iy++;
        console_printf(iy++, 0, 5, 0, " -- Timing (avg %d frames) -----------------------------------", fc);
        console_printf(iy++, 0, 5, 0, "  Kernel exec:  %8.1f us  (GPU hw: %7.1f us, queue: %7.1f us)",
                       avg, gpu_k, gpu_q);
        console_printf(iy++, 0, 5, 0, "  Result copy:  %8.1f us", t_rb);
        console_printf(iy++, 0, 5, 0, "  Frame total:  %8.1f us  |  %.1f fps", frame_us, 1e6 / frame_us);
        iy++;
        console_printf(iy++, 0, 4, 0, " Arrows=move  q=quit  r=restart  |  Grid %dx%d on GPU", gw, gh);

        if (status[0] == 2) {
            console_printf(iy + 2, 0, 2, 1, " GAME OVER! ");
            console_printf(iy + 3, 0, 4, 0, " r=restart  q=quit");
        }

        console_flush();
#else
        erase();
        attron(A_BOLD | COLOR_PAIR(6));
        mvprintw(0, 0, " GPU SNAKE %dx%d=%d cells | %s %s %dCU@%dMHz ",
                 gw, gh, gw * gh, gpu.platform_name, gpu.device_name,
                 gpu.compute_units, gpu.clock_freq);
        attroff(A_BOLD | COLOR_PAIR(6));

        for (int y = 0; y < gh; y++) {
            move(y + 1, 0);
            for (int x = 0; x < gw; x++) {
                int v = grid[y * gw + x];
                int c, a;
                if (y == head[0] && x == head[1]) { c = '@'; a = COLOR_PAIR(3) | A_BOLD; }
                else if (v == -1) { c = '*'; a = COLOR_PAIR(2) | A_BOLD; }
                else if (v > 0) { c = 'o'; a = COLOR_PAIR(1); }
                else { c = ' '; a = 0; }
                addch(c | a);
            }
        }

        int iy = lay.info_y;
        char buf[512];

        attron(COLOR_PAIR(4) | A_BOLD);
        snprintf(buf, sizeof(buf), " Score:%d  Length:%d  Frame:%d  Pos(%d,%d)  FPS:%.1f",
                 len[0]-4, len[0], fc, head[1], head[0], fps);
        mvaddstr(iy++, 0, buf);
        attroff(COLOR_PAIR(4) | A_BOLD);

        iy++;
        attron(COLOR_PAIR(5));
        mvaddstr(iy++, 0, " -- GPU -----------------------------------------------------");
        snprintf(buf, sizeof(buf), "  Platform:   %s", gpu.platform_name);
        mvaddstr(iy++, 0, buf);
        snprintf(buf, sizeof(buf), "  Device:     %s  (%s)", gpu.device_name, gpu.is_gpu ? "GPU" : "CPU");
        mvaddstr(iy++, 0, buf);
        snprintf(buf, sizeof(buf), "  CUs: %d  Freq: %d MHz  Grid: %dx%d  Threads: %d",
                 gpu.compute_units, gpu.clock_freq, gw, gh, gw * gh);
        mvaddstr(iy++, 0, buf);
        iy++;
        snprintf(buf, sizeof(buf), " -- Timing (avg %d frames) -----------------------------------", fc);
        mvaddstr(iy++, 0, buf);
        snprintf(buf, sizeof(buf), "  Kernel exec:  %8.1f us  (GPU hw: %7.1f us, queue: %7.1f us)",
                 avg, gpu_k, gpu_q);
        mvaddstr(iy++, 0, buf);
        snprintf(buf, sizeof(buf), "  Result copy:  %8.1f us", t_rb);
        mvaddstr(iy++, 0, buf);
        snprintf(buf, sizeof(buf), "  Frame total:  %8.1f us  |  %.1f fps", frame_us, 1e6 / frame_us);
        mvaddstr(iy++, 0, buf);
        attroff(COLOR_PAIR(5));

        iy++;
        attron(COLOR_PAIR(4));
        snprintf(buf, sizeof(buf), " Arrows=move  q=quit  r=restart  |  Grid %dx%d on GPU", gw, gh);
        mvaddstr(iy++, 0, buf);
        attroff(COLOR_PAIR(4));

        if (status[0] == 2) {
            attron(A_BOLD | COLOR_PAIR(2));
            mvaddstr(iy + 1, 0, " GAME OVER! ");
            attroff(A_BOLD | COLOR_PAIR(2));
            mvaddstr(iy + 2, 0, " r=restart  q=quit");
        }

        refresh();
#endif
        continue;

    restart:
        init_game(grid, head, dir, len, status, rng, gw, gh);
        gpu_upload(&gpu, grid, head, dir, len, status, rng, gw, gh);
        fc = 0; t_sum = 0; sess = now_us();
        status[0] = 0;
    }

quit:
    gpu_free(&gpu);
#ifdef USE_WINCONSOLE
    console_cleanup();
#else
    endwin();
#endif
    free(grid);
    printf("Device: %s %s\n", gpu.platform_name, gpu.device_name);
    printf("Grid: %dx%d, %d frames\n", gw, gh, fc);
    return 0;
}
