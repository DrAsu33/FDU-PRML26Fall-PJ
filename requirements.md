# PRML 课程项目要求：Post-training a General LM for Chess

> **负责助教**：葛煦旸
> **项目周期与截止时间**：`2026/11/17 ~ 2026/12/26`（完成 PJ 项目并提交课程论文）

---

## 1. 项目目标与评分要点

- **核心任务**：以 `Qwen3.5-0.8B` 或 `Qwen3.5-2B` 为基座模型，通过 SFT 或 RL（强化学习）训练一个会下国际象棋的语言模型。
- **评分要点**：最终成绩由 **棋力评价** 与 **实验报告质量** 综合决定。
  - **棋力评价**：通过统一对局 harness 在限定算力（单卡 RTX 4090）和限时（每步 15 秒）下进行实战评测。
  - **报告质量**：需清晰说明训练方案、数据来源与用量（特别是使用外部大模型或棋类专用模型生成数据时）、实验设计与复现方式。

---

## 2. 基座模型与训练要求

| 项目 | 具体要求 |
| :--- | :--- |
| **基座模型** | **仅限** `Qwen3.5-0.8B` 或 `Qwen3.5-2B`（Base 或 Instruct 版本均可），尺寸可根据算力和训练方法自行权衡。 |
| **权重限制** | **严禁**使用他人已在国际象棋上微调过的权重。 |
| **训练数据** | 不限。可使用 Lichess 对局库、局面评估库、谜题库及其他任意公开数据。 |
| **合成数据** | 允许使用更大的 LLM 或棋类专用模型生成训练数据，但**必须在报告中说明来源与用量**。 |
| **训练方法** | 不限，可自行决定是否包含 RL 阶段。 |

---

## 3. 对局 Harness（`player.py`）与评测规则

每人需提交一个 Python 脚本 `player.py` 作为对局 harness，实现以下接口：

```python
import chess

class Player:
    def __init__(self, model_dir: str):
        """加载 model_dir 下的模型与分词器。"""

    def act(self, board: chess.Board) -> chess.Move:
        """返回当前局面下要走的着法。着法历史可从 board.move_stack 获取。"""
```

### 评测环境与约束

- **允许自定义的内容**：提示格式（Prompt）、解码参数、是否多次采样、是否让模型自我校验等均可自行决定。
- **依赖白名单**：脚本**仅允许依赖** `python-chess`、`PyTorch`、`transformers`、`vLLM` 以及你提交的模型。
- **严禁行为**：不得调用象棋引擎（如 Stockfish）、其他模型、网络服务，或使用开局库、残局库。
- **硬件与时间限制**：运行环境为**单张 RTX 4090（24 GB 显存）**，**每步限时 15 秒**。
- **判负规则**：某一步超时、抛出异常或返回非法着法，**当局直接判负**（*提示：可以在 `player.py` 中自行实现合法性检查与重试逻辑*）。

---

## 4. 提交内容清单

最终提交需包含以下 4 个部分：

- [ ] `model/`：HuggingFace 格式的完整模型权重（若使用 LoRA 微调，**必须先合并权重**）。
- [ ] `player.py`：符合上述接口规范的对局 harness 脚本。
- [ ] `code/`：完整的数据处理与训练代码，**需附复现说明**。
- [ ] `report.pdf`：实验报告（需包含数据来源与用量、方法、实验结果与分析等）。

---

## 5. 云端算力与集群使用说明

课程为每组提供 1 个**开发容器**和 1 个同名 Linux 账号 `gNN`（如 `g07`，组内共用）。开发容器用于写代码、管理数据和提交任务，**本身无 GPU**；需要 GPU 的任务需通过 **Slurm** 提交至计算节点运行。

- **集群总资源**：32 张 RTX 4090（24 GB）。
- **每组配额**：总预算 **350 卡时**，同一时刻最多同时占用 **2 张 GPU**。

### 5.1 配置 SSH 连接

1. 在本地电脑生成 SSH 密钥（若已有可跳过）：
   ```bash
   ssh-keygen -t ed25519
   cat ~/.ssh/id_ed25519.pub
   ```
2. 将输出的公钥（以 `ssh-ed25519` 开头的一行）填写至课程分组多维表格。
3. 助教分配组号后，使用以下命令连接开发容器（将 `gNN` 替换为实际组号）：
   ```bash
   ssh -p 20018 gNN@180.163.52.129
   ```

### 5.2 开发容器环境

| 项目 | 说明 |
| :--- | :--- |
| **硬件资源** | 4 核 CPU、16 GB 内存，无 GPU |
| **`/home/gNN`** | 组目录，**300 GB 配额**，代码、环境、数据和 checkpoint 均存放于此 |
| **`/shared`** | 只读目录，存放课程预置的 base 模型权重 |
| **预装软件** | Ubuntu 22.04、Python 3.10、`uv`、`git`、`tmux`、`zsh` |
| **镜像源** | `pip`/`uv` 默认使用阿里云 PyPI 镜像；`HF_ENDPOINT` 已配置为 `hf-mirror.com` |

> **重要提醒**：
> - 无 `root` 权限，无法使用 `apt install`。Python 依赖请通过 `uv` 或 `venv` 安装在组目录下：
>   ```bash
>   uv venv && source .venv/bin/activate
>   uv pip install torch transformers
>   ```
> - 只有 `/home/gNN` 目录会被持久化保留；容器重启后，运行中的进程和 `tmux` 会话都会丢失。
> - **集群数据无备份**，重要实验结果和权重请及时下载到本地备份。

### 5.3 使用 GPU（Slurm 调度）

Slurm 任务运行在与开发容器一致的环境中：可访问组目录和 `/shared`，当前工作目录及已激活的虚拟环境会自动带入任务。

#### 分区与配额限制

| 分区 | 单任务 GPU 上限 | 时长上限 | 用途 |
| :--- | :--- | :--- | :--- |
| `train`（默认） | ≤ 2 | 24 小时 | 正式训练 |
| `debug` | 1 | 1 小时 | 调试与快速验证（排队优先） |

- **并发上限**：每组所有任务合计最多占用 **2 张 GPU**。
- **默认 CPU / 内存**：每张 GPU 默认分配 **12 核 CPU**，每核约 3.7 GB 内存（单卡约 **46 GB 内存**），可用 `-c`、`--mem` 调整。
- **预算与 `-t` 参数（关键）**：每组总 Quota 为 **350 卡时**。
  - **请务必用 `-t` 指定合理的运行时长**。任务仅在剩余预算 ≥「GPU 数 × 申请时长」时才会启动；若不写 `-t`，将按分区上限计算（例如 2 卡任务需剩余至少 48 卡时才能启动）。
  - 实际扣费按**实际运行时间**计算。合理缩短 `-t` 更利于调度器利用空闲时段提前插空运行。

#### 常用提交与管理命令

```bash
# 1. 交互式运行（等待分配，输出实时打印到终端）
srun -p debug --gres=gpu:1 -t 10 nvidia-smi
srun -p debug --gres=gpu:1 -t 30 python eval.py

# 2. 后台提交正式训练（断开 SSH 不影响）
sbatch --gres=gpu:2 -t 8:00:00 -o logs/train-%j.log --wrap "python train.py"

# 3. 申请 GPU 节点交互式 Shell 调试
srun -p debug --gres=gpu:1 -t 30 --pty zsh

# 4. 查看与管理任务
squeue --me                                                        # 查看自己的任务及排队原因
scancel <jobid>                                                    # 取消指定任务
sacct -X -o JobID,JobName,Partition,AllocTRES%30,Elapsed,State     # 查看历史任务记录
scontrol show assoc_mgr flags=qos qos=budget_$USER | grep -o 'gres/gpu=[^,]*'  # 查看预算：总额(已用)，单位 GPU·分钟
```

#### 排队规则与常见状态

- `debug` 分区优先于 `train` 获取空闲 GPU，但不会抢占正在运行的任务。
- 同一分区内，近期用量较少的组优先级更高（用量每 3 天减半），排队越久优先级越高；较短任务更容易触发回填（backfill）提前运行。
- `squeue --me` 常见排队原因：
  - `Resources`：集群空闲 GPU 不足
  - `Priority`：前方有更高优先级的任务
  - `AssocGrpGRES`：本组已达到 2 张卡的并发上限
  - `QOSGrpGRESMinutes`：剩余预算不足以支付当前申请的「GPU 数 × 时长」

#### 任务内文件系统

- 任务根文件系统为**只读**，可写路径包括：组目录（`/home/gNN`）、`/tmp`、`/scratch`。
- `/scratch` 为计算节点本地 NVMe，读写极快，适合存放临时文件；但**仅在当前任务内可见、各节点不互通、任务结束后不保留**。

### 5.4 共享基座模型与资源下载

`/shared/models` 目录下已预置各选题基座模型（HuggingFace 格式），**可直接加载，请勿重复下载**：

- `/shared/models/Qwen3.5-0.8B`（本项目可选）
- `/shared/models/Qwen3.5-0.8B-Base`（本项目可选）
- `/shared/models/Qwen3.5-2B`（本项目可选）
- `/shared/models/Qwen3.5-2B-Base`（本项目可选）
- `/shared/models/Qwen2.5-VL-3B-Instruct`
- `/shared/models/Qwen3-1.7B`
- `/shared/models/Qwen3-4B`

加载示例：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/shared/models/Qwen3.5-0.8B", torch_dtype="auto", device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("/shared/models/Qwen3.5-0.8B")
```

如需下载其他公开数据集或辅助模型到组目录，推荐使用 ModelScope 或 HuggingFace CLI：

```bash
uvx modelscope download --model Qwen/Qwen3-8B --local_dir ~/models/Qwen3-8B
uvx --from huggingface_hub hf download <repo> --local-dir ~/models/<name>
```

---

## 6. 其他注意事项

1. **禁止在开发容器跑重型 CPU 任务**：大规模数据预处理等耗时 CPU 任务请提交至 Slurm，例如：
   ```bash
   srun -c 8 -t 1:00:00 python preprocess.py
   ```
2. **权限隔离**：每组仅可访问自己的组目录，请勿尝试访问其他组的数据。
3. **求助规范**：遇到集群或环境问题联系助教时，请一并附上**组号、任务 ID（JobID）和完整报错信息**。
