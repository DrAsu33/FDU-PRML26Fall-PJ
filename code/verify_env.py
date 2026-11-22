"""一键验证本地（WSL/Linux 或 Windows）Stockfish 与 Qwen3.5-2B 部署是否正常。"""

import os
from pathlib import Path
import shutil
import sys
import time
import chess
import chess.engine
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from player import Player  # noqa: E402


def find_stockfish_binary(stockfish_dir: Path) -> Path:
    if stockfish_dir.exists():
        if os.name == "nt":
            candidates = sorted(stockfish_dir.rglob("stockfish*.exe"))
        else:
            candidates = sorted(
                p
                for p in stockfish_dir.rglob("stockfish*")
                if p.is_file()
                and not p.name.endswith((".exe", ".zip", ".tar", ".gz", ".md", ".txt"))
                and os.access(p, os.X_OK)
            )
            # 若目录内放了软链接或未加 +x，放宽匹配仅名为 stockfish 的文件
            if not candidates:
                candidates = sorted(
                    p
                    for p in stockfish_dir.rglob("stockfish*")
                    if p.is_file() and os.access(p, os.X_OK)
                )
        if candidates:
            return candidates[0]

    which_sf = shutil.which("stockfish")
    if which_sf:
        return Path(which_sf)

    raise FileNotFoundError(
        f"未在 {stockfish_dir} 或系统 PATH 中找到可执行 Stockfish 引擎。"
    )


def main() -> None:
    skip_model = "--skip-model" in sys.argv
    model_dir = ROOT / "model" / "Qwen3.5-2B"
    stockfish_dir = ROOT / "Stockfish"

    print(f"[1/3] 检查 GPU: CUDA={torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()

    sf_path = find_stockfish_binary(stockfish_dir)
    print(f"[2/3] 检查 Stockfish: {sf_path}")
    engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    print(f"      引擎名称: {engine.id.get('name', 'unknown')}")

    board = chess.Board()
    info = engine.analyse(board, chess.engine.Limit(time=0.1))
    print(f"      初始局面评估: {info['score'].white()} | 推荐着法: {info['pv'][0].uci()}")

    if skip_model:
        engine.quit()
        print("\nGPU 与 Stockfish 验证通过（已跳过模型加载）！")
        return

    print(f"[3/3] 加载模型并测试对弈: {model_dir}")
    t0 = time.monotonic()
    player = Player(str(model_dir))
    load_sec = time.monotonic() - t0
    vram_gb = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    )
    print(f"      加载完成: {load_sec:.2f}s | 峰值显存: {vram_gb:.2f} GB")

    # 让 Qwen3.5-2B (白方) 与低难度 Stockfish (黑方) 试下 4 个回合（8 步）
    engine.configure({"Skill Level": 0})
    for ply in range(8):
        if board.is_game_over():
            break
        if board.turn == chess.WHITE:
            t1 = time.monotonic()
            move = player.act(board)
            dt = time.monotonic() - t1
            assert move in board.legal_moves, f"非法着法: {move}"
            print(f"      第 {ply + 1} 步 [Qwen3.5-2B 白]: {move.uci()} (耗时 {dt:.2f}s)")
        else:
            res = engine.play(board, chess.engine.Limit(time=0.1))
            move = res.move
            print(f"      第 {ply + 1} 步 [Stockfish  黑]: {move.uci()}")
        board.push(move)

    engine.quit()
    print("\n部署验证全部通过！")


if __name__ == "__main__":
    main()
