import re
import time
import chess
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

UCI_PATTERN = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")


def format_ascii_board(board: chess.Board) -> str:
    """将当前棋盘展开为带行列坐标的 8x8 字符棋盘，方便大语言模型对齐行列。"""
    fen_board = board.fen().split(" ")[0]
    rows = fen_board.split("/")
    lines = []
    for idx, row in enumerate(rows):
        rank = 8 - idx
        expanded = []
        for ch in row:
            if ch.isdigit():
                expanded.extend("." * int(ch))
            else:
                expanded.append(ch)
        lines.append(f"{rank} | " + " ".join(expanded))
    lines.append("  +----------------")
    lines.append("    a b c d e f g h")
    return "\n".join(lines)


def build_chess_prompt(board: chess.Board, legal_ucis: list[str]) -> str:
    """项目统一棋盘 Prompt 格式（v1 标准）。训练与推理必须严格保持一致。"""
    fen = board.fen()
    fen_parts = fen.split(" ")
    side = "White" if board.turn == chess.WHITE else "Black"
    castling = fen_parts[2] if len(fen_parts) > 2 else "-"
    en_passant = fen_parts[3] if len(fen_parts) > 3 else "-"
    recent_moves = " ".join(m.uci() for m in board.move_stack[-16:]) or "(none)"

    return (
        "You are a grandmaster chess player. Choose the single best legal move in UCI format.\n\n"
        f"Board:\n{format_ascii_board(board)}\n\n"
        f"FEN: {fen}\n"
        f"Side to move: {side}\n"
        f"Castling: {castling}\n"
        f"En passant: {en_passant}\n"
        f"Recent moves (UCI): {recent_moves}\n"
        f"Legal moves (UCI): {' '.join(legal_ucis)}\n\n"
        "Output ONLY the best move in UCI format (e.g. e2e4), nothing else."
    )


class Player:
    def __init__(self, model_dir: str):
        """加载 model_dir 下的模型与分词器。"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Qwen3.5 为原生多模态架构，优先用 AutoModelForImageTextToText 加载，失败回退 AutoModelForCausalLM
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_dir,
                dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
        self.model.eval()

    def _render_prompt(self, user_msg: str) -> str:
        """若为 Instruct 模型则应用对话模板并关闭思考模式；Base 模型直接拼接续写后缀。"""
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": user_msg}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return user_msg + "\nBest move: "

    def _parse_legal_move(self, text: str, legal_set: set[str]) -> chess.Move | None:
        """从模型输出文本中提取合法的 UCI 着法。"""
        for match in UCI_PATTERN.findall(text.lower()):
            if match in legal_set:
                return chess.Move.from_uci(match)
        return None

    def act(self, board: chess.Board) -> chess.Move:
        """返回当前局面下要走的着法。着法历史可从 board.move_stack 获取。"""
        start_time = time.monotonic()
        legal_moves = list(board.legal_moves)
        legal_ucis = sorted(m.uci() for m in legal_moves)
        legal_set = set(legal_ucis)

        user_msg = build_chess_prompt(board, legal_ucis)
        prompt = self._render_prompt(user_msg)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        # 最多尝试 3 次（首次贪心解码，失败后采样重试），并预留时间防止超过 15 秒上限
        for attempt in range(3):
            if time.monotonic() - start_time > 12.0:
                break
            try:
                with torch.no_grad():
                    gen_kwargs = dict(
                        max_new_tokens=16,
                        do_sample=(attempt > 0),
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    if attempt > 0:
                        gen_kwargs["temperature"] = 0.7
                    output_ids = self.model.generate(**inputs, **gen_kwargs)
                generated = self.tokenizer.decode(
                    output_ids[0][input_len:], skip_special_tokens=True
                ).strip()
                move = self._parse_legal_move(generated, legal_set)
                if move is not None:
                    return move
            except Exception:
                continue

        # 兜底：若模型未能生成合法着法，返回字典序第一个合法着法，确保绝不判负
        return chess.Move.from_uci(legal_ucis[0])
