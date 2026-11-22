# PRML Chess LM — 本地部署与开发指南

首版计划用 `Qwen3.5-2B` 做监督微调（SFT），在相同对局条件下与微调前的基座比较棋力。课程也允许使用 `Qwen3.5-0.8B`；仅做 SFT 即符合训练方法要求，是否加入强化学习（RL）可根据实验结果和剩余卡时决定。
完整课程要求见 [requirements.md](./requirements.md)，AI Agent 协作规范见 [AGENTS.md](./AGENTS.md)。

---

## 1. 文件体积与目录结构

本地需要部署的两个核心组件体积都很可控：

| 组件 | 存放目录 | 体积大小 | 用途说明 |
| :--- | :--- | :--- | :--- |
| **Stockfish 引擎** | `Stockfish/` | 约 **78 MB**（解压后 ~103 MB） | 开源象棋引擎（Stockfish 19），**仅用于数据标注与离线棋力评测**（严禁在 `player.py` 中调用） |
| **Qwen3.5-2B 基座** | `model/Qwen3.5-2B/` | 约 **4.55 GB** | 项目基座模型（BF16 推理峰值显存实测仅 **4.43 GB**，RTX 4090 Laptop 16 GB 轻松运行） |

目录布局：

```text
PRML-PJ/
├── model/                  # 模型权重目录（不上传 GitHub；本地放基座/微调权重，最终交作业时放合并后的最终模型）
│   └── Qwen3.5-2B/
├── Stockfish/              # Stockfish 象棋引擎目录（不上传 GitHub；仅用于离线评测与造数据）
│   └── stockfish/          # 解压后的可执行文件（Linux/WSL 无后缀或 Windows .exe）
├── code/                   # 数据处理、训练、评测与环境验证脚本（上传 GitHub）
│   └── verify_env.py       # 一键验证 GPU、Stockfish 与 Qwen3.5-2B 是否正常工作
├── player.py               # 对局 harness（上传 GitHub）
├── requirements.md         # 课程项目要求（上传 GitHub）
├── AGENTS.md               # AI Agent 协作与一键部署规范（上传 GitHub）
├── .gitignore              # Git 忽略规则（上传 GitHub）
└── README.md               # 本说明文档（上传 GitHub）
```

> **重要提醒**：`model/` 和 `Stockfish/` 均已写入 `.gitignore`，不会被误提交到 Git 仓库。最终提交作业时，需将训练并合并完成后的最终模型权重放入 `model/`，且不要将 `Stockfish/` 打包进 `player.py` 的推理依赖。

---

## 2. 棋盘与着法表示（首版默认 v1）

课程允许自行设计 Prompt。v1 是首版实验的默认格式。同一模型在训练、验证和对局时使用相同的棋盘表示、聊天模板、生成前缀和着法格式。后续实验可以调整字段或比较不同格式，在实验记录中注明模型对应的版本。

### 2.1 输入格式（Prompt）

下面是从初始局面走完 `e2e4` 后，传给模型的用户消息正文。复现时先取 `legal_ucis = sorted(m.uci() for m in board.legal_moves)`，再调用 `player.py` 中的 `build_chess_prompt(board, legal_ucis)`。

```text
You are a grandmaster chess player. Choose the single best legal move in UCI format.

Board:
8 | r n b q k b n r
7 | p p p p p p p p
6 | . . . . . . . .
5 | . . . . . . . .
4 | . . . . P . . .
3 | . . . . . . . .
2 | P P P P . P P P
1 | R N B Q K B N R
  +----------------
    a b c d e f g h

FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
Side to move: Black
Castling: KQkq
En passant: -
Recent moves (UCI): e2e4
Legal moves (UCI): a7a5 a7a6 b7b5 b7b6 b8a6 b8c6 c7c5 c7c6 d7d5 d7d6 e7e5 e7e6 f7f5 f7f6 g7g5 g7g6 g8f6 g8h6 h7h5 h7h6

Output ONLY the best move in UCI format (e.g. e2e4), nothing else.
```

**设计原因**：
1. **展开的 8×8 棋盘**：把原版 FEN 里的数字缩写（如 `8`）展开成 `.`，并标上 `a–h` 列和 `1–8` 行（大写 = 白子 `PNBRQK`，小写 = 黑子 `pnbrqk`），让大语言模型能直接对齐行列位置，不必做隐式算术计数。
2. **FEN 与历史着法**。FEN 提供当前局面及行棋方、易位权等信息。有棋谱时，回放到目标着法之前，保留最近最多 16 个半回合（一方走一次算一个半回合）。只有 FEN 的局面也可用于训练，历史字段统一填 `Recent moves (UCI): (none)`。
3. **合法着法列表**。`python-chess` 从当前局面生成候选着法，按 UCI 字典序列出，由模型选择其中一步。固定顺序使同一局面的候选列表可复现。

### 2.2 输出格式（Move）

- **纯小写 UCI 格式**：`起点格 + 终点格 + 可选升变字母`，如 `e2e4`、`e1g1`（王车易位）、`e7e8q`（兵升变后）。
- 目标文本只含着法。Instruct 模型通过 `enable_thinking=False` 关闭思考模式，单步耗时以单张 RTX 4090 上的对局实测为准。

---

## 3. 本地环境与依赖安装（推荐 WSL2）

前置条件：WSL2（Ubuntu 22.04/24.04）+ Python 3.10+ + NVIDIA 驱动（RTX 4090 Laptop 16 GB 显存）。

推荐使用 `.venv` + `uv` 搭配**清华源**安装依赖（Linux PyPI 版 `torch` 默认自带 CUDA 支持；`Qwen3.5` 需 `transformers >= 5.2.0`）：

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip uv -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv/bin/uv pip install torch transformers accelerate python-chess modelscope \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

激活环境并检查 CUDA 是否可用：

```bash
source .venv/bin/activate
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

---

## 4. 部署 Stockfish（放到 `Stockfish/` 文件夹）

Stockfish 官方已提供通用版二进制（自动适配 AVX2 / AVX-512，无需手动区分 CPU 指令集）：

### WSL / Linux 一键下载并解压（推荐）

```bash
mkdir -p Stockfish
curl -L -C - -f --retry 5 -o Stockfish/stockfish.tar.gz \
  https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-linux-x86-64-universal.tar.gz
tar -xzf Stockfish/stockfish.tar.gz -C Stockfish
rm Stockfish/stockfish.tar.gz
```

### Windows PowerShell（备选）

```powershell
New-Item -ItemType Directory -Force -Path Stockfish
curl.exe -L -f -o Stockfish/stockfish.zip https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-universal.zip
Expand-Archive -Path Stockfish/stockfish.zip -DestinationPath Stockfish -Force
Remove-Item Stockfish/stockfish.zip
```

*(如果 GitHub 下载较慢，也可以手动从 [Stockfish 官网下载页](https://stockfishchess.org/download/) 下载对应平台压缩包并解压到 `Stockfish/` 目录下。)*

---

## 5. 部署 Qwen3.5-2B 大模型（放到 `model/` 文件夹）

> **下载提速提醒**：请按步骤**逐个下载**（不要同时跑多个大文件下载挤占带宽）；国内网络如开启了本地代理，下载国内镜像源（清华源 / ModelScope）时加上 `NO_PROXY="*" HTTP_PROXY="" HTTPS_PROXY=""` 可直连跑满带宽。

### 场景 A：本地 WSL / Linux（通过 ModelScope 下载，推荐）

```bash
source .venv/bin/activate
mkdir -p model
NO_PROXY="*" HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy="" \
  modelscope download --model Qwen/Qwen3.5-2B --local_dir ./model/Qwen3.5-2B
```

### 场景 B：学校云端集群（禁止重复下载，直接软链接共享模型）

```bash
mkdir -p model
ln -sfn /shared/models/Qwen3.5-2B ./model/Qwen3.5-2B
```

### 备选：HuggingFace 国内镜像下载

```bash
HF_ENDPOINT=https://hf-mirror.com \
  hf download Qwen/Qwen3.5-2B --local-dir ./model/Qwen3.5-2B
```

---

## 6. 一键验证部署结果

所有组件部署完成后，运行验证脚本检查 GPU、Stockfish 引擎、显存占用、单步延迟，并让 `Qwen3.5-2B` 与 Stockfish 实战试下 8 步：

```bash
source .venv/bin/activate
python code/verify_env.py
# 如果只想先测 GPU 和 Stockfish（跳过大模型加载）：
python code/verify_env.py --skip-model
```

---

## 7. Git 版本控制规范（GitHub 同步说明）

本项目已配置 [.gitignore](./.gitignore)，推送到 GitHub 时严格区分**追踪文件**与**忽略文件**：

- **应该被 Git 追踪（提交到 GitHub）**：
  - 源代码：`player.py`、`code/` 下的所有训练/数据处理/评测/验证脚本
  - 文档与配置：`README.md`、`AGENTS.md`、`requirements.md`、`.gitignore`、`pyproject.toml` / `requirements.txt`（后续添加）
  - 轻量级评测结果摘要：如小体积 JSON/CSV 指标汇总表、报告图表（供写 `report.pdf` 使用）
- **不要被 Git 追踪（已在 `.gitignore` 中屏蔽）**：
  - 模型权重与断点：`model/`、`checkpoints/`、`*.safetensors`、`*.bin`、`*.pt`、`*.pth`
  - 象棋引擎二进制：`Stockfish/`
  - 大规模棋谱与数据集：`data/`、`*.pgn`、`*.zst`、`*.jsonl`、`*.parquet`、各类压缩包
  - 本地虚拟环境与缓存：`.venv/`、`__pycache__/`、`logs/`、`wandb/`
  - 密钥与敏感配置：`.env`、`id_ed25519*`、`*.key`

在新机器（或云端集群）拉取仓库后的标准复现流程：
```bash
git clone <你的仓库地址> PRML-PJ && cd PRML-PJ
# 然后按本文档第 3 → 4 → 5 → 6 节依次部署（或直接对 AI Agent 说“一键部署”）
```
