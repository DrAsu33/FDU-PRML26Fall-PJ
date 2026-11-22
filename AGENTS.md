# AGENTS.md — PRML Chess LM 项目协作指南

本文件面向在本仓库中工作的 AI Agent，定义了项目的硬性约束、交付结构、本地与云端执行规范，以及日常开发守则。

---

## 1. 项目概览与硬性约束（Hard Constraints）

- **项目目标**：以 `Qwen3.5-0.8B` 或 `Qwen3.5-2B`（Base 或 Instruct 版本均可）为基座，通过 SFT / RL 训练会下国际象棋的语言模型。
- **首版实验**默认用 `Qwen3.5-2B` 做 SFT，并与微调前的同一基座比较棋力。课程允许仅做 SFT；是否加入 RL 由实验结果和算力预算决定。课程对基座、评测和提交的具体要求见 [requirements.md](./requirements.md)。
- **截止时间**：`2026/12/26`。
- **最终交付结构**：
  - `model/`：HuggingFace 格式完整模型权重（**若使用 LoRA，最终必须合并为完整权重**）。
  - `player.py`：对局 harness 脚本。
  - `code/`：数据处理与训练脚本，附清晰的复现文档（README / 运行命令）。
  - `report.pdf`：实验报告（需记录所有训练数据来源、合成数据所用模型及用量、训练配方与评测结果）。

### 绝对红线（严禁违反）

1. **基座限制**：仅可使用 `Qwen3.5-0.8B` / `Qwen3.5-0.8B-Base` / `Qwen3.5-2B` / `Qwen3.5-2B-Base`。**严禁**加载任何他人已在国际象棋上微调过的权重。
2. **`player.py` 推理白名单**：
   - 仅允许依赖：`python-chess`、`torch`、`transformers`、`vllm` 以及提交的 `model/`。
   - **严禁**在 `player.py` 中调用象棋引擎（如 Stockfish）、其他外部模型、任何网络请求、开局库或残局库。
3. **`player.py` 接口与容错要求**：
   - 必须精确保持以下类与方法签名：
     ```python
     import chess

     class Player:
         def __init__(self, model_dir: str): ...
         def act(self, board: chess.Board) -> chess.Move: ...
     ```
   - 评测环境为**单张 RTX 4090（24 GB 显存）**，**每步限时 15 秒**。
   - 超时、抛异常或返回非法着法均直接判负。**实现 `act()` 时必须包含时间预算控制、合法着法校验（`move in board.legal_moves`）、重试机制以及兜底合法着法**，确保绝不因非法着法或超时判负。

---

## 2. 棋盘与着法表示（首版默认 v1）

> 课程允许自行设计 Prompt。对同一模型，训练、验证和对局推理须使用同一版本的输入与着法格式，包括聊天模板、生成前缀及特殊 token。后续实验可比较其他表示方式，并在实验记录中注明各 checkpoint 对应的格式版本。

### 输入格式（Prompt）
v1 输入包含带坐标的 8×8 字符棋盘、FEN、局面元信息、最近最多 16 个半回合的 UCI 历史，以及按字典序排列的合法着法列表。

- 展开棋盘用 `.` 表示空格，大写为白子（`P N B R Q K`），小写为黑子（`p n b r q k`）。
- 有棋谱时，回放到**目标着法之前**，取最近最多 16 个半回合（一方走一次算一个半回合）。只有 FEN 的样本也可使用：以 `chess.Board(fen)` 构造局面，历史字段按现有实现写作 `Recent moves (UCI): (none)`。
- 先取 `legal_ucis = sorted(m.uci() for m in board.legal_moves)`，再调用 `player.py` 的 `build_chess_prompt(board, legal_ucis)`。该函数内部调用 `format_ascii_board(board)`。列表只提供当前局面的候选着法，由模型选出一步。

示例：
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

### 输出格式（Target）
- **纯小写 UCI 着法**：匹配 `^[a-h][1-8][a-h][1-8][qrbn]?$`（如 `e2e4`、`e1g1`、`e7e8q`），无多余解释文本。
- Instruct 模型使用 `apply_chat_template(..., enable_thinking=False)` 关闭思考模式。按第 5 节的单卡实测检查每步 15 秒的限制。

---

## 3. 本地运行指南（Local Execution）

- **本地环境**：WSL2（Ubuntu，工作目录 `/home/rog/Mycode/PRML-PJ`），RTX 4090 Laptop（16 GB 显存）。
- **本地目录约定**：
  - `.venv/`：本地 Python 虚拟环境（通过 `uv` + 清华源安装 `torch`、`transformers >= 5.2.0`、`accelerate`、`python-chess`、`modelscope`）。
  - `model/`：存放本地基座模型（如 `model/Qwen3.5-2B/`）及训练产物（最终提交时仅保留合并后的完整权重）。
  - `Stockfish/`：存放本地 Linux 原生版 Stockfish 引擎，**仅用于数据标注与离线棋力评测，严禁在 `player.py` 中调用**。
- **Qwen3.5 加载注意事项**：
  - `Qwen3.5-0.8B` / `Qwen3.5-2B` 为原生多模态架构（`qwen3_5`），需要 `transformers >= 5.2.0`，加载时优先使用 `AutoModelForImageTextToText`（兼容回退 `AutoModelForCausalLM`）。

### 3.1 一键部署指令规范（当用户输入“一键部署”时）

当用户在新克隆的仓库（本地 WSL/Linux 或云端集群）中输入**“一键部署”**时，Agent 必须**按以下顺序串行执行**（切勿并发执行多个大下载任务，以免挤爆带宽导致 TLS 断连）：

1. **环境与平台探测**：
   - 检查操作系统、Python 版本、磁盘剩余空间与 GPU（`nvidia-smi`）。
   - 检测是否存在 `/shared/models/Qwen3.5-2B`：若存在，判定为**学校云端集群**；否则判定为**本地 WSL / Linux**。
2. **创建并安装 `.venv` 虚拟环境**：
   - 执行 `python3 -m venv .venv`（或 `uv venv`），优先使用**清华 PyPI 镜像**（`https://pypi.tuna.tsinghua.edu.cn/simple`，国内网络可加 `NO_PROXY="*"` 避免代理绕路）：
     ```bash
     python3 -m venv .venv
     .venv/bin/pip install -U pip uv -i https://pypi.tuna.tsinghua.edu.cn/simple
     .venv/bin/uv pip install torch transformers accelerate python-chess modelscope \
       --index-url https://pypi.tuna.tsinghua.edu.cn/simple
     ```
   - 确认 `transformers >= 5.2.0` 且（在 GPU 节点上）`torch.cuda.is_available() == True`。
3. **部署 Stockfish 到 `Stockfish/`**（若尚未存在）：
   - 下载官方 Linux 通用版（支持 `-C -` 断点续传与重试）并解压：
     ```bash
     mkdir -p Stockfish
     curl -L -C - -f --retry 5 -o Stockfish/stockfish.tar.gz \
       https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-linux-x86-64-universal.tar.gz
     tar -xzf Stockfish/stockfish.tar.gz -C Stockfish && rm Stockfish/stockfish.tar.gz
     ```
4. **部署 `Qwen3.5-2B` 到 `model/Qwen3.5-2B/`**（若尚未存在）：
   - **云端集群**：严禁重复下载，直接建立软链接 `mkdir -p model && ln -sfn /shared/models/Qwen3.5-2B model/Qwen3.5-2B`。
   - **本地 WSL / Linux**：绕过代理直连 ModelScope 断点续传下载：
     ```bash
     mkdir -p model
     NO_PROXY="*" HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy="" \
       .venv/bin/modelscope download --model Qwen/Qwen3.5-2B --local_dir ./model/Qwen3.5-2B
     ```
5. **端到端验证**：
   - 本地 GPU 环境直接运行 `.venv/bin/python code/verify_env.py`（若在无 GPU 的云端开发容器内，改用 `srun -p debug --gres=gpu:1 -t 10 .venv/bin/python code/verify_env.py`），并向用户汇报 GPU 识别情况、Stockfish 版本、模型加载耗时、峰值显存和单步推理延迟。
- **适合在本地完成的任务**：
  - 编写、重构与审查代码（数据处理、训练脚本、评测脚本、`player.py`）。
  - 轻量级单元测试与逻辑验证（例如：使用 Mock 模型验证 `player.py` 的 Prompt 构造、UCI/SAN 解析、非法着法重试、超时保护、边界局面如升变/王车易位/吃过路兵）。
  - 小样本数据抽查、日志分析、绘图及撰写实验报告。
  - Git 版本控制与从云端同步备份重要 checkpoint / 评测结果（云端无备份）。
- **本地运行注意事项**：
  - 优先使用 `uv` 管理 Python 虚拟环境，避免污染系统环境。
  - **不要**在本地执行大规模数据下载/清洗或正式模型训练，重计算任务统一放到云端 Slurm 集群运行。
  - 所有脚本路径应使用跨平台写法（`pathlib.Path`），避免硬编码 Windows 反斜杠路径导致云端 Linux 报错。

---

## 4. 云端运行指南（Cloud / Cluster Execution）

### 4.1 SSH 连接与文件同步（待生成）

> **[TODO / 待生成]** 分组账号与 SSH 密钥配置尚未录入。待用户提供组号与本机公钥配置后，在此补全以下信息：
> - **组号 / 账号**：`gNN`（待填写）
> - **主机与端口**：`180.163.52.129`，端口 `20018`
> - **SSH 登录命令**：`ssh -p 20018 gNN@180.163.52.129`（待替换为实际组号或 SSH Config 别名）
> - **SSH Config 别名**：`待生成`（如 `~/.ssh/config` 中的 `Host prml-cluster`）
> - **代码同步命令（本地 → 云端）**：`待生成`（如 `rsync` / `scp` / `git push + pull`）
> - **结果回传命令（云端 → 本地备份）**：`待生成`

### 4.2 开发容器规范（Login / Dev Container）

- **配置**：4 核 CPU、16 GB RAM、**无 GPU**；Ubuntu 22.04、Python 3.10、`uv`、`git`、`tmux`、`zsh`。
- **持久化与配额**：
  - 组目录 `/home/gNN` 配额 **300 GB**，是唯一持久化目录；代码、虚拟环境、数据、checkpoint 均须放于此。
  - 容器重启会清除运行中的进程和 `tmux` 会话；且**无数据备份**，重要 checkpoint 与评测报告必须及时同步回本地。
- **依赖管理**：
  - 无 `root` 权限，**不能**执行 `apt install`。
  - 使用 `uv venv && source .venv/bin/activate` + `uv pip install ...` 安装依赖（已预配阿里云 PyPI 镜像和 `HF_ENDPOINT=hf-mirror.com`）。
- **禁止行为**：**切勿**在开发容器直接跑重 CPU 任务（如大规模 PGN 解析/数据预处理），此类任务必须提交给 Slurm CPU 任务运行：
  ```bash
  srun -c 8 -t 1:00:00 python preprocess.py
  ```

### 4.3 Slurm GPU 计算规范

- **算力预算与并发限制**：
  - 总预算 **350 卡时**，同一时刻最多占用 **2 张 RTX 4090（24 GB）**。
  - 每张 GPU 默认配 12 核 CPU、约 46 GB 内存（可用 `-c`、`--mem` 调整）。
- **分区选择**：
  - `debug`：最多 1 卡、最长 1 小时、排队优先 → 用于环境验证、小步数 smoke test、单卡 `player.py` 评测。
  - `train`（默认）：最多 2 卡、最长 24 小时 → 用于正式 SFT / RL 训练。
- **必须显式设置合理的 `-t` 时长**：
  - 调度器要求剩余预算 ≥ `GPU数 × 申请时长` 才放行任务（不写 `-t` 会按上限扣预留，2 卡默认需 48 卡时）。
  - 实际按运行时间扣费；`-t` 越紧凑越容易触发回填（backfill）提前启动。
- **任务内文件系统**：
  - 根文件系统**只读**；仅 `/home/gNN`、`/tmp`、`/scratch` 可写。
  - `/scratch` 是节点本地 NVMe，适合高频读写临时文件（如解压中间数据），但任务结束后销毁且跨节点不共享；最终 checkpoint 必须写回 `/home/gNN`。
- **预置共享模型（禁止重复下载）**：
  - 直接从 `/shared/models/` 加载：
    - `/shared/models/Qwen3.5-0.8B`
    - `/shared/models/Qwen3.5-0.8B-Base`
    - `/shared/models/Qwen3.5-2B`
    - `/shared/models/Qwen3.5-2B-Base`
  - 额外数据集/辅助模型下载到 `/home/gNN` 时优先使用 `uvx modelscope download` 或 `uvx --from huggingface_hub hf download`。

### 4.4 常用 Slurm 命令速查

```bash
# 快速调试 / 单卡验证（debug 分区）
srun -p debug --gres=gpu:1 -t 15 nvidia-smi
srun -p debug --gres=gpu:1 -t 30 python eval.py
srun -p debug --gres=gpu:1 -t 30 --pty zsh

# 后台正式训练（train 分区，日志落盘，断开 SSH 不中断）
sbatch --gres=gpu:2 -t 6:00:00 -o logs/train-%j.log --wrap "python train.py"

# 监控与配额查询
squeue --me                                                        # 查看排队状态与原因
scancel <jobid>                                                    # 取消任务
sacct -X -o JobID,JobName,Partition,AllocTRES%30,Elapsed,State     # 查看历史任务
scontrol show assoc_mgr flags=qos qos=budget_$USER | grep -o 'gres/gpu=[^,]*'  # 剩余预算（GPU·分钟）
```

---

## 5. Agent 工作流与工程守则

1. **先验证再放量**：任何训练脚本上 `train` 分区前，先在本地做语法/单测检查，再在云端 `debug` 分区用极小数据跑通 1–2 个 step 的 smoke test，确认显存、吞吐与 checkpoint 保存正常后再提交长时任务，避免浪费 350 卡时配额。
2. **记录可追溯信息**：所有实验脚本应记录数据来源、样本数量、合成数据生成方式、关键超参数与 Slurm JobID，方便直接汇总到 `report.pdf` 和 `code/` 的复现说明中。
3. **关注推理端兼容性**：训练时的 Prompt 模板、特殊 token、着法表示格式（UCI / SAN）必须与 `player.py` 严格一致；每次产出新模型后，务必在单卡 RTX 4090 上端到端测试 `player.py` 的加载显存、单步延迟（严格 < 15s）与合法率。
4. **交付前检查**：若训练采用 LoRA/QLoRA，务必执行权重合并并验证合并后的 `model/` 能直接被 `Player(model_dir)` 加载运行。
5. **Git 仓库卫生（严禁提交大文件与密钥）**：
   - **必须追踪**：`player.py`、`code/`、`README.md`、`AGENTS.md`、`requirements.md`、`.gitignore` 及轻量级配置/评测汇总结果。
   - **严禁追踪**（由 `.gitignore` 屏蔽）：`model/`、`checkpoints/`、`Stockfish/`、`data/`、`.venv/`、`logs/`、`wandb/`、各类模型权重（`*.safetensors`/`*.bin`/`*.pt`）、大规模棋谱压缩包（`*.pgn`/`*.zst`/`*.jsonl`）以及 SSH 私钥/环境变量（`.env`/`id_ed25519*`）。执行 `git push` 前务必先检查 `git status`。
