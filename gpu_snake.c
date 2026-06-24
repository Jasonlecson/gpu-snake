#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <ncurses.h>
#include <locale.h>
#include <CL/cl.h>

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

typedef struct {
    int grid_w, grid_h;
    int map_h;
    int info_y;
} layout_t;

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void check_cl(cl_int err, const char *msg) {
    if (err != CL_SUCCESS) {
        endwin();
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
        endwin();
        fprintf(stderr, "No OpenCL platforms found\n");
        exit(1);
    }

    cl_platform_id *platforms = malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);

    // try GPU first, then any device
    g.device = NULL;
    for (cl_uint p = 0; p < num_platforms && !g.device; p++) {
        cl_uint num_devs;
        if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devs) == CL_SUCCESS && num_devs > 0) {
            cl_device_id *devs = malloc(num_devs * sizeof(cl_device_id));
            clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, num_devs, devs, NULL);
            g.device = devs[0];
            g.platform = platforms[p];
            g.is_gpu = 1;
            free(devs);
        }
    }
    if (!g.device) {
        for (cl_uint p = 0; p < num_platforms && !g.device; p++) {
            cl_uint num_devs;
            if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devs) == CL_SUCCESS && num_devs > 0) {
                cl_device_id *devs = malloc(num_devs * sizeof(cl_device_id));
                clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devs, devs, NULL);
                g.device = devs[0];
                g.platform = platforms[p];
                g.is_gpu = 0;
                free(devs);
            }
        }
    }
    free(platforms);

    if (!g.device) {
        endwin();
        fprintf(stderr, "No OpenCL devices found\n");
        exit(1);
    }

    clGetPlatformInfo(g.platform, CL_PLATFORM_NAME, sizeof(g.platform_name), g.platform_name, NULL);
    clGetDeviceInfo(g.device, CL_DEVICE_NAME, sizeof(g.device_name), g.device_name, NULL);
    clGetDeviceInfo(g.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &g.compute_units, NULL);
    clGetDeviceInfo(g.device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(int), &g.clock_freq, NULL);

    g.ctx = clCreateContext(NULL, 1, &g.device, NULL, NULL, &err);
    check_cl(err, "ctx");
    g.queue = clCreateCommandQueue(g.ctx, g.device, CL_QUEUE_PROFILING_ENABLE, &err);
    check_cl(err, "queue");

    g.prog = clCreateProgramWithSource(g.ctx, 1, &KERNEL_SRC, NULL, &err);
    check_cl(err, "prog");
    err = clBuildProgram(g.prog, 1, &g.device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz;
        clGetProgramBuildInfo(g.prog, g.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char *log = malloc(log_sz);
        clGetProgramBuildInfo(g.prog, g.device, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
        endwin();
        fprintf(stderr, "Build:\n%s\n", log);
        free(log);
        exit(1);
    }

    g.kernel = clCreateKernel(g.prog, "snake_update", &err);
    check_cl(err, "kernel");
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
    clEnqueueWriteBuffer(g->queue, g->grid_g, CL_TRUE, 0, w * h * sizeof(int), grid, 0, NULL, NULL);
    clEnqueueWriteBuffer(g->queue, g->head_g, CL_TRUE, 0, 2 * sizeof(int), head, 0, NULL, NULL);
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

static void calc_layout(layout_t *l) {
    struct winsize ws;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws);
    int sw = ws.ws_col;
    int sh = ws.ws_row;
    if (sw < 20) sw = 20;
    if (sh < 10) sh = 10;

    int info = sh / 4;
    if (info < 8) info = 8;
    l->info_y = sh - info;
    l->map_h = l->info_y - 1;
    l->grid_w = sw - 2;
    l->grid_h = l->map_h - 1;
    if (l->grid_w < 5) l->grid_w = 5;
    if (l->grid_h < 3) l->grid_h = 3;
}

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

int main(void) {
    srand(time(NULL));
    gpu_ctx_t gpu = gpu_init();
    layout_t lay;
    calc_layout(&lay);

    int gw = lay.grid_w, gh = lay.grid_h;
    int *grid = calloc(gw * gh, sizeof(int));
    int head[2], dir[1], len[1], status[1], rng[1];
    init_game(grid, head, dir, len, status, rng, gw, gh);
    gpu_alloc(&gpu, gw, gh, grid);
    gpu_upload(&gpu, grid, head, dir, len, status, rng, gw, gh);

    setlocale(LC_ALL, "");
    initscr(); cbreak(); noecho(); curs_set(0);
    nodelay(stdscr, TRUE); timeout(80); keypad(stdscr, TRUE);
    start_color(); use_default_colors();
    init_pair(1, COLOR_GREEN, -1);
    init_pair(2, COLOR_RED, -1);
    init_pair(3, COLOR_YELLOW, -1);
    init_pair(4, COLOR_WHITE, -1);
    init_pair(5, COLOR_CYAN, -1);
    init_pair(6, COLOR_MAGENTA, -1);
    init_pair(7, COLOR_BLUE, -1);

    double t_sum = 0;
    int fc = 0;
    double sess = now_us();

    while (1) {
        double tf = now_us();

        int ch = getch();
        switch (ch) {
            case KEY_UP:    if (dir[0] != 1) dir[0] = 0; break;
            case KEY_DOWN:  if (dir[0] != 0) dir[0] = 1; break;
            case KEY_LEFT:  if (dir[0] != 3) dir[0] = 2; break;
            case KEY_RIGHT: if (dir[0] != 2) dir[0] = 3; break;
            case 'q': goto quit;
            case 'r': goto restart;
        }

        // resize check
        layout_t nl;
        calc_layout(&nl);
        if (nl.grid_w != gw || nl.grid_h != gh) {
            int old_w = gw, old_h = gh;
            gw = nl.grid_w; gh = nl.grid_h;
            lay = nl;
            int *ng = calloc(gw * gh, sizeof(int));
            // copy old grid content
            for (int y = 0; y < (gh < old_h ? gh : old_h); y++)
                for (int x = 0; x < (gw < old_w ? gw : old_w); x++)
                    ng[y * gw + x] = grid[y * old_w + x];
            free(grid);
            grid = ng;
            // clamp head
            if (head[0] >= gh) head[0] = gh - 1;
            if (head[1] >= gw) head[1] = gw - 1;
            // realloc GPU buffers
            clReleaseMemObject(gpu.grid_g);
            cl_int err;
            gpu.grid_g = clCreateBuffer(gpu.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        gw * gh * sizeof(int), grid, &err);
            fc = 0; t_sum = 0; sess = now_us();
        }

        // dir copy
        clEnqueueWriteBuffer(gpu.queue, gpu.dir_g, CL_TRUE, 0, sizeof(int), dir, 0, NULL, NULL);

        // kernel
        double t0 = now_us();
        size_t gws = gw * gh;
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

        // read back
        t0 = now_us();
        clEnqueueReadBuffer(gpu.queue, gpu.head_g, CL_TRUE, 0, 2*sizeof(int), head, 0, NULL, NULL);
        clEnqueueReadBuffer(gpu.queue, gpu.status_g, CL_TRUE, 0, sizeof(int), status, 0, NULL, NULL);
        clEnqueueReadBuffer(gpu.queue, gpu.len_g, CL_TRUE, 0, sizeof(int), len, 0, NULL, NULL);
        clEnqueueReadBuffer(gpu.queue, gpu.grid_g, CL_TRUE, 0, gw*gh*sizeof(int), grid, 0, NULL, NULL);
        double t_rb = now_us() - t0;

        // GPU profiling
        cl_ulong ps, pt, pe;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_SUBMIT, sizeof(ps), &ps, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(pt), &pt, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(pe), &pe, NULL);
        double gpu_q = (pt - ps) / 1e3;
        double gpu_k = (pe - pt) / 1e3;
        clReleaseEvent(ev);

        fc++;
        t_sum += t_kernel;

        // render
        erase();
        attron(A_BOLD | COLOR_PAIR(6));
        mvprintw(0, 0, " GPU SNAKE %dx%d=%d cells | %s %s %dCU@%dMHz ",
                 gw, gh, gw * gh,
                 gpu.platform_name, gpu.device_name, gpu.compute_units, gpu.clock_freq);
        int title_len = 40 + strlen(gpu.platform_name) + strlen(gpu.device_name);
        for (int i = title_len; i < lay.grid_w + 2; i++) addch(' ');
        attroff(A_BOLD | COLOR_PAIR(6));

        for (int y = 0; y < gh; y++) {
            move(y + 1, 0);
            for (int x = 0; x < gw && x < lay.grid_w; x++) {
                int v = grid[y * gw + x];
                int c, a;
                if (y == head[0] && x == head[1]) {
                    c = '@'; a = COLOR_PAIR(3) | A_BOLD;
                } else if (v == -1) {
                    c = '*'; a = COLOR_PAIR(2) | A_BOLD;
                } else if (v > 0) {
                    c = 'o'; a = COLOR_PAIR(1);
                } else {
                    c = ' '; a = 0;
                }
                addch(c | a);
            }
        }

        // info
        int iy = lay.info_y;
        double elapsed = (now_us() - sess) / 1e6;
        double fps = fc / elapsed;
        double avg = t_sum / fc;
        double frame_us = now_us() - tf;
        char buf[512];

        attron(COLOR_PAIR(4) | A_BOLD);
        snprintf(buf, sizeof(buf), " Score:%d  Length:%d  Frame:%d  Pos(%d,%d)  FPS:%.1f",
                 len[0]-4, len[0], fc, head[1], head[0], fps);
        mvaddstr(iy++, 0, buf);
        attroff(COLOR_PAIR(4) | A_BOLD);

        iy++;
        attron(COLOR_PAIR(5));
        mvaddstr(iy++, 0, " \xe2\x94\x80\xe2\x94\x80 GPU \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80");
        snprintf(buf, sizeof(buf), "  Platform:   %s", gpu.platform_name);
        mvaddstr(iy++, 0, buf);
        snprintf(buf, sizeof(buf), "  Device:     %s  (%s)", gpu.device_name, gpu.is_gpu ? "GPU" : "CPU/Other");
        mvaddstr(iy++, 0, buf);
        snprintf(buf, sizeof(buf), "  CUs: %d  Freq: %d MHz  Grid: %dx%d  Threads: %d",
                 gpu.compute_units, gpu.clock_freq, gw, gh, gw * gh);
        mvaddstr(iy++, 0, buf);
        iy++;
        snprintf(buf, sizeof(buf), " \xe2\x94\x80\xe2\x94\x80 Timing (avg %d frames) \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80", fc);
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
        snprintf(buf, sizeof(buf), " Arrows=move  q=quit  r=restart  |  Terminal=Grid %dx%d", gw, gh);
        mvaddstr(iy++, 0, buf);
        attroff(COLOR_PAIR(4));

        if (status[0] == 2) {
            attron(A_BOLD | A_BLINK | COLOR_PAIR(2));
            mvaddstr(iy + 1, 0, " GAME OVER! ");
            attroff(A_BOLD | A_BLINK | COLOR_PAIR(2));
            mvaddstr(iy + 2, 0, " r=restart  q=quit");
        }

        refresh();
        continue;

    restart:
        init_game(grid, head, dir, len, status, rng, gw, gh);
        gpu_upload(&gpu, grid, head, dir, len, status, rng, gw, gh);
        fc = 0; t_sum = 0; sess = now_us();
        status[0] = 0;
    }

quit:
    gpu_free(&gpu);
    endwin();
    free(grid);
    printf("Device: %s %s\n", gpu.platform_name, gpu.device_name);
    printf("Grid: %dx%d, %d frames\n", gw, gh, fc);
    return 0;
}
