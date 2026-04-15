"""
chess_engine.py
---------------
Stockfish 엔진 연동 전담 모듈.

담당 기능:
  - 엔진 시작 / 종료 / 크래시 시 자동 재시작
  - 분석 전 보드 유효성 검사 (빈 보드, 킹 없음 등)
  - 최선 수 분석 (move + score)
  - 평가 점수 포맷
  - 캡처 여부 판단 헬퍼
"""

from dataclasses import dataclass
from typing import Optional

import chess
import chess.engine

import config


# ─────────────────────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────────────────────
@dataclass
class AnalysisResult:
    """분석 결과 컨테이너."""
    move:     chess.Move
    uci:      str          # "e2e4"
    san:      str          # "e4"
    move_str: str          # "E2 → E4  (e4)"
    eval_str: str          # "+0.35"  또는  "Mate in 3"
    score:    Optional[chess.engine.PovScore] = None


# ─────────────────────────────────────────────────────────────
# 엔진 래퍼
# ─────────────────────────────────────────────────────────────
class ChessEngine:
    """
    Stockfish 엔진 래퍼.
    - 크래시(SIGSEGV 등) 감지 후 자동 재시작
    - 분석 전 보드 유효성 검사로 빈 보드/킹 없음 크래시 방지

    Parameters
    ----------
    path      : stockfish 실행 파일 경로
    think_time: 탐색 시간 (초)
    """

    def __init__(
        self,
        path:       str   = config.STOCKFISH_PATH,
        think_time: float = config.THINK_TIME,
    ):
        self._path      = path
        self.think_time = think_time
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._restart()

    # ── 시작 / 재시작 ────────────────────────────────────────
    def _restart(self) -> None:
        """엔진 (재)시작. 기존 프로세스는 강제 종료 후 새로 띄움."""
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:
                pass
            self._engine = None

        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(self._path)
            print(f"[Engine] Stockfish 시작: {self._path}")
        except FileNotFoundError:
            raise RuntimeError(
                f"Stockfish 없음: {self._path}\n"
                "  → sudo apt install stockfish  또는 config.py 경로 수정"
            )

    # ── 보드 유효성 검사 ─────────────────────────────────────
    @staticmethod
    def validate_board(board: chess.Board) -> str:
        """
        Stockfish에 넘기기 전 보드 상태 검사.
        문제 있으면 오류 메시지 반환, 없으면 빈 문자열 반환.
        """
        # 양쪽 킹이 모두 있어야 함
        if not board.pieces(chess.KING, chess.WHITE):
            return "백 킹이 없습니다 — 보드 스캔을 먼저 실행하세요"
        if not board.pieces(chess.KING, chess.BLACK):
            return "흑 킹이 없습니다 — 보드 스캔을 먼저 실행하세요"
        # 합법적인 수가 있어야 함
        if board.is_game_over():
            return f"게임 종료 상태입니다 ({board.result()})"
        if not any(board.legal_moves):
            return "합법적인 수가 없습니다 (스테일메이트?)"
        return ""

    # ── 공개 API ────────────────────────────────────────────
    def analyse(self, board: chess.Board) -> AnalysisResult:
        """
        현재 보드를 분석해 최선 수와 평가 점수 반환.

        Raises
        ------
        RuntimeError : 보드 유효성 오류, 추천 수 없음, 엔진 크래시
        """
        # 보드 유효성 검사
        err = self.validate_board(board)
        if err:
            raise RuntimeError(err)

        # 엔진 크래시 감지 → 자동 재시작 후 재시도
        for attempt in range(2):
            if self._engine is None:
                self._restart()
            try:
                result = self._engine.analyse(
                    board,
                    chess.engine.Limit(time=self.think_time),
                )
                break
            except chess.engine.EngineTerminatedError:
                if attempt == 0:
                    print("[Engine] 크래시 감지 → 재시작 중...")
                    self._restart()
                else:
                    raise RuntimeError("Stockfish 재시작 후에도 실패")
            except chess.engine.EngineError as e:
                if attempt == 0:
                    print(f"[Engine] 오류({e}) → 재시작 중...")
                    self._restart()
                else:
                    raise RuntimeError(f"Stockfish 오류: {e}")

        move = result.get("pv", [None])[0]
        if move is None:
            raise RuntimeError("추천 수 없음 (게임 종료?)")

        score    = result.get("score")
        uci      = move.uci()
        san      = board.san(move)
        move_str = f"{uci[:2].upper()} → {uci[2:].upper()}  ({san})"
        eval_str = self._format_score(score, board.turn)

        return AnalysisResult(
            move=move,
            uci=uci,
            san=san,
            move_str=move_str,
            eval_str=eval_str,
            score=score,
        )

    def quit(self) -> None:
        """엔진 종료."""
        if self._engine:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    @property
    def is_ready(self) -> bool:
        return self._engine is not None

    # ── 내부 헬퍼 ───────────────────────────────────────────
    @staticmethod
    def _format_score(
        score: Optional[chess.engine.PovScore],
        turn:  chess.Color,
    ) -> str:
        """평가 점수를 사람이 읽기 쉬운 문자열로 변환."""
        if score is None:
            return ""
        pov = score.white()
        if pov.is_mate():
            return f"Mate in {pov.mate()}"
        cp = pov.score()
        if cp is None:
            return ""
        side = "백" if turn == chess.WHITE else "흑"
        sign = "+" if cp > 0 else ""
        return f"{sign}{cp / 100:.2f}  ({side} 차례)"


# ─────────────────────────────────────────────────────────────
# 체스 유틸
# ─────────────────────────────────────────────────────────────
def is_capture(board: chess.Board, move: chess.Move) -> bool:
    """해당 수가 캡처 수인지 반환."""
    return board.is_capture(move)


def is_promotion(move: chess.Move) -> bool:
    """해당 수가 프로모션인지 반환."""
    return move.promotion is not None


def label_to_piece(label: str) -> Optional[chess.Piece]:
    """비전 레이블 → python-chess Piece 변환."""
    _MAP = {
        "wP": chess.Piece(chess.PAWN,   chess.WHITE),
        "wN": chess.Piece(chess.KNIGHT, chess.WHITE),
        "wB": chess.Piece(chess.BISHOP, chess.WHITE),
        "wR": chess.Piece(chess.ROOK,   chess.WHITE),
        "wQ": chess.Piece(chess.QUEEN,  chess.WHITE),
        "wK": chess.Piece(chess.KING,   chess.WHITE),
        "bP": chess.Piece(chess.PAWN,   chess.BLACK),
        "bN": chess.Piece(chess.KNIGHT, chess.BLACK),
        "bB": chess.Piece(chess.BISHOP, chess.BLACK),
        "bR": chess.Piece(chess.ROOK,   chess.BLACK),
        "bQ": chess.Piece(chess.QUEEN,  chess.BLACK),
        "bK": chess.Piece(chess.KING,   chess.BLACK),
    }
    return _MAP.get(label)


def board_from_vision(piece_map: dict) -> chess.Board:
    """비전 get_board() 결과 → python-chess Board 변환."""
    board = chess.Board(fen=None)   # 빈 보드
    for sq_name, label in piece_map.items():
        piece = label_to_piece(label)
        if piece:
            board.set_piece_at(chess.parse_square(sq_name), piece)
    return board