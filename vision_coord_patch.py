"""
vision_coord_patch.py
---------------------
DualDobotController에 비전 좌표를 연결하는 모듈.

YOLO 모드:
  말의 실제 픽셀 위치 → 캘리브 행렬로 mm 변환
  탐지 실패 시 → square_to_mm_for() 실측값으로 fallback

칸 중심 이동:
  square_to_mm_for() 실측값 직접 사용 (픽셀 변환 없음 — 가장 정확)
"""

from typing import Callable, Optional

import chess

from dual_dobot_controller import (
    DualDobotController,
    Robot,
    square_to_mm_for,
    ROW_A_MIN, ROW_A_MAX,
    RANKS,
)
from vision_coord import VisionRobotCalib
import config


class VisionGuidedController(DualDobotController):

    def __init__(
        self,
        vision,
        calib: VisionRobotCalib,
        log_fn: Callable = print,
        use_vision: bool = True,
    ):
        super().__init__(log_fn=log_fn)
        self.vision     = vision
        self.calib      = calib
        self.use_vision = use_vision

    # ── 로봇A mm → 로봇B mm 변환 ─────────────────────────────
    def _convert_to_robot_b(self, x_mm_a: float, y_mm_a: float) -> tuple:
        """
        로봇A 기준 mm → 로봇B 기준 mm.
        square_to_mm_for() 실측 공식 역산 사용.

        로봇A 공식:
            x = ORIGIN_X_MM + (7 - row) * (CELL_A - 2.0)
            y = ORIGIN_Y_MM + col * (CELL_A + 0.3)

        로봇B 공식:
            x = ORIGIN_B_X + row * CELL_B
            y = ORIGIN_B_Y + (7 - col) * CELL_B
        """
        _CELL_A = config.BOARD_MM_A / 8
        _CELL_B = config.BOARD_MM_B / 8

        # 로봇A mm → row/col 역산
        row = 7 - (x_mm_a - config.ORIGIN_X_MM) / (_CELL_A - 2.0)
        col = (y_mm_a - config.ORIGIN_Y_MM) / (_CELL_A + 0.3)

        # row/col → 로봇B mm
        x_mm_b = config.ORIGIN_B_X + row * _CELL_B
        y_mm_b = config.ORIGIN_B_Y + (7 - col) * _CELL_B

        return (round(x_mm_b, 1), round(y_mm_b, 1))

    # ── 칸 중심 mm — 실측값 직접 사용 ───────────────────────
    def _get_center_mm(self, square: str, robot: Robot) -> tuple:
        """실측 square_to_mm_for() 값 사용 — 가장 정확."""
        x_mm, y_mm = square_to_mm_for(Robot.A, square)
        if robot == Robot.B:
            x_mm, y_mm = self._convert_to_robot_b(x_mm, y_mm)
        return (round(x_mm, 1), round(y_mm, 1))

    # ── 집기 위치 mm ─────────────────────────────────────────
    def _get_pick_mm(
        self,
        square: str,
        robot: Robot,
        expected_label: Optional[str] = None,
    ) -> tuple:
        """
        YOLO로 말 실제 위치 탐지 → mm 변환.
        탐지 실패 or 비전 없으면 실측 칸 중심 사용.
        """
        if not self.use_vision or not self.calib.is_ready:
            return square_to_mm_for(robot, square)

        if self.vision.yolo_mode:
            px = self.vision.get_piece_pixel(square)
            if px is not None:
                wx, wy = px
                x_mm, y_mm = self.calib.pixel_to_mm(wx, wy)
                if robot == Robot.B:
                    x_mm, y_mm = self._convert_to_robot_b(x_mm, y_mm)
                    x_mm += config.YOLO_PICK_OFFSET_B_X
                    y_mm += config.YOLO_PICK_OFFSET_B_Y
                else:
                    x_mm += config.YOLO_PICK_OFFSET_A_X
                    y_mm += config.YOLO_PICK_OFFSET_A_Y
                self.log(f"[YOLO] {square.upper()} 실제위치 픽셀({wx:.0f},{wy:.0f}) → ({x_mm:.1f},{y_mm:.1f})mm", "info")
                return (round(x_mm, 1), round(y_mm, 1))
            else:
                self.log(f"[YOLO] {square.upper()} 탐지 실패 → 칸 중심으로 이동", "warn")

        return self._get_center_mm(square, robot)

    # ── 놓기 위치 mm ─────────────────────────────────────────
    def _get_place_mm(self, square: str, robot: Robot) -> tuple:
        return square_to_mm_for(robot, square)

    # ── execute_move 오버라이드 ───────────────────────────────
    def execute_move(self, board: chess.Board, move: chess.Move) -> None:
        from_sq  = chess.square_name(move.from_square)
        to_sq    = chess.square_name(move.to_square)
        from_row = RANKS.index(from_sq[1])
        to_row   = RANKS.index(to_sq[1])

        piece = board.piece_at(move.from_square)
        expected_label = _piece_to_label(piece) if piece else None

        # ① 캡처 처리
        if board.is_capture(move):
            cap_sq = to_sq
            if board.is_en_passant(move):
                ep_rank = "5" if board.turn == chess.WHITE else "4"
                cap_sq  = to_sq[0] + ep_rank

            cap_piece = board.piece_at(chess.parse_square(cap_sq))
            cap_label = _piece_to_label(cap_piece) if cap_piece else None
            cap_row   = RANKS.index(cap_sq[1])
            cap_robot = Robot.A if ROW_A_MIN <= cap_row <= ROW_A_MAX else Robot.B

            cap_mm = self._get_pick_mm(cap_sq, cap_robot, cap_label)
            gy_mm  = self._next_graveyard(board.turn)

            self.log(f"[Vision] 캡처: {cap_sq.upper()} mm={cap_mm}", "info")
            self._park_other(self._other(cap_robot))
            self._pick_and_place_rs(
                self._get_rs(cap_robot), cap_mm, gy_mm,
                label=f"캡처({cap_sq.upper()})"
            )

        # ② 말 이동
        worker = self._select_robot(from_row, to_row)

        if worker is None:
            picker = Robot.A if from_row >= ROW_A_MIN else Robot.B
            placer = self._other(picker)
            from_mm = self._get_pick_mm(from_sq, picker, expected_label)
            to_mm   = self._get_place_mm(to_sq, placer)
            self.log(f"[Vision] Handoff: {from_sq.upper()}→{to_sq.upper()}", "info")
            self._handoff_move(from_mm, to_mm, picker, placer)
        else:
            from_mm = self._get_pick_mm(from_sq, worker, expected_label)
            to_mm   = self._get_place_mm(to_sq, worker)
            self.log(
                f"[Vision] {from_sq.upper()}→{to_sq.upper()} "
                f"Robot {worker.value} from={from_mm} to={to_mm}", "info"
            )
            self._park_other(self._other(worker))
            self._pick_and_place_rs(
                self._get_rs(worker), from_mm, to_mm,
                label=f"{from_sq.upper()}→{to_sq.upper()}"
            )

        self.log("[Vision] 수 실행 완료", "ok")

    # ── GUI 캐슬링용 외부 접근 ───────────────────────────────
    def _vision_pick_mm(self, square: str, robot: Robot, expected_label=None):
        return self._get_pick_mm(square, robot, expected_label)

    def _vision_place_mm(self, square: str, robot: Robot):
        return self._get_place_mm(square, robot)


def _piece_to_label(piece: chess.Piece) -> str:
    color = "w" if piece.color == chess.WHITE else "b"
    sym   = chess.piece_symbol(piece.piece_type).upper()
    return color + sym