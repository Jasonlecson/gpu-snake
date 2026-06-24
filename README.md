# GPU Snake

贪吃蛇游戏，核心逻辑在 GPU 上通过 OpenCL 并行执行。网格大小自适应终端窗口。

![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 功能

- **GPU 计算** — 蛇的移动、碰撞检测、食物生成全部在 GPU 上并行执行
- **自适应网格** — 终端多大，格子就是多少
- **实时性能指标** — 显示 GPU kernel 执行时间、队列延迟、FPS 等
- **跨平台** — Linux、macOS 使用 ncurses，Windows 使用原生 Console API
- **零依赖** — Windows 版单个 exe，不需要任何 DLL

## 下载

从 [Releases](https://github.com/Jasonlecson/gpu-snake/releases) 下载对应平台的二进制文件：

| 平台 | 文件 | 说明 |
|------|------|------|
| Linux x86_64 | `gpu_snake-linux-x86_64` | `chmod +x` 后直接运行 |
| macOS x86_64 | `gpu_snake-macos-x86_64` | `chmod +x` 后直接运行 |
| Windows x86_64 | `gpu_snake-windows-x86_64.exe` | 双击或命令行运行 |

**前置条件：** 系统需安装 GPU 驱动（AMD / NVIDIA / Intel），驱动自带 OpenCL 运行时。

## 从源码编译

### Linux

```bash
# Debian/Ubuntu
sudo apt install build-essential libncurses-dev ocl-icd-opencl-dev opencl-headers

# Fedora
sudo dnf install gcc ncurses-devel ocl-icd-devel opencl-headers

# Arch
sudo pacman -S gcc ncurses ocl-icd opencl-headers

# 编译运行
make
./gpu_snake
```

如果遇到 GPU 权限问题：

```bash
sudo usermod -aG render $USER   # 重新登录生效
# 或临时使用：
sg render -c "./gpu_snake"
```

### macOS

```bash
xcode-select --install
make
./gpu_snake
```

### Windows

使用 [MSYS2](https://www.msys2.org) 的 MinGW64 环境：

```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-opencl-icd mingw-w64-x86_64-opencl-headers
make
./gpu_snake.exe
```

Windows 版使用原生 Console API，不依赖 ncurses，编译后为单个 exe。

## 操作

| 按键 | 功能 |
|------|------|
| 方向键 | 移动 |
| `q` | 退出 |
| `r` | 重新开始 |

## 工作原理

每帧，GPU 执行一个 kernel 并行处理所有格子：

```
┌─────────────────────────────────────────────────┐
│  GPU Kernel (所有格子并行)                        │
│                                                  │
│  1. Aging    — 蛇身格子 age++，超龄变空           │
│  2. barrier  — 同步，等待所有格子更新完毕          │
│  3. Move     — 线程 0 计算蛇头新位置              │
│  4. Collision — 检测撞墙/撞自己                   │
│  5. Food     — 吃到食物则随机生成新食物            │
└─────────────────────────────────────────────────┘
         ↓
    CPU 读回结果 → 终端渲染
```

## GPU 自动检测

按以下顺序搜索 OpenCL 设备：

1. 遍历所有平台，优先使用 **GPU 设备**（AMD / NVIDIA / Intel）
2. 找不到 GPU 则回退到任意 OpenCL 设备（如 pocl CPU）

无需手动配置，有 OpenCL 运行时就能跑。

## 项目结构

```
gpu-snake-game/
├── gpu_snake.c              # 源码（跨平台，含 OpenCL kernel）
├── Makefile                 # 编译脚本（自动检测平台）
├── LICENSE                  # MIT 许可证
├── README.md
├── .gitignore
└── .github/
    └── workflows/
        └── release.yml      # GitHub Actions 自动构建三平台二进制
```

## 许可证

[MIT](LICENSE)
