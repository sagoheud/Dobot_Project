"""
dual_dobot_controller.py
------------------------
도봇 2대 협력 제어 모듈.

■ 담당 열
  로봇 A (왼쪽)  : a ~ e 열  (col 0~4)
  로봇 B (오른쪽) : d ~ h 열  (col 3~7)
  겹치는 d~e 열 → 이동 중점 X 기준 가까운 쪽 우선

■ 좌표계
  로봇A : a1 = ORIGIN_X_MM, ORIGIN_Y_MM  /  a→h=+X, 1→8=+Y
  로봇B : h8 = ORIGIN_B_X,  ORIGIN_B_Y   /  col·row 반전 (마주보는 배치)

■ 이동 순서
  1) 캡처 수 → 상대 말 묘지로 먼저 이동
  2) 단일 로봇 처리 또는 handoff
  3) 비작업 로봇은 대기점으로 복귀 (충돌 방지)

■ 경유점 vs 대기점
  경유점(WAYPOINT): 픽앤플레이스 시작 전/후 경유 — 관절 안전 구간 확보
  대기점(HOME)    : 작업 대기 위치 — 팔이 체스판 바깥을 향한 자세

■ 비상 정지
  request_abort() → 큐 정지 → 그리퍼 off → 대기점 복귀
"""

import math
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

import chess
from pydobot import Dobot
from pydobot.message import Message
from pydobot.enums.CommunicationProtocolIDs import CommunicationProtocolIDs
from pydobot.enums.ControlValues import ControlValues

import config

# ─────────────────────────────────────────────────────────────
# config 값
# ─────────────────────────────────────────────────────────────
DOBOT_PORT_A = config.DOBOT_PORT_A
DOBOT_PORT_B = config.DOBOT_PORT_B

HOME_A_X = config.HOME_A_X
HOME_A_Y = config.HOME_A_Y
HOME_A_R = config.HOME_A_R
HOME_B_X = config.HOME_B_X
HOME_B_Y = config.HOME_B_Y
HOME_B_R = config.HOME_B_R

WAYPOINT_X = config.WAYPOINT_X
WAYPOINT_Y = config.WAYPOINT_Y
WAYPOINT_R = config.WAYPOINT_R

HANDOFF_A_X = config.HANDOFF_A_X
HANDOFF_A_Y = config.HANDOFF_A_Y
HANDOFF_B_X = config.HANDOFF_B_X
HANDOFF_B_Y = config.HANDOFF_B_Y
GRAVEYARD_A = config.GRAVEYARD_A
GRAVEYARD_B = config.GRAVEYARD_B

# 담당 행 범위 (rank 인덱스: 1=0 … 8=7)
ROW_A_MIN, ROW_A_MAX = 3, 7  # rank 4~8
ROW_B_MIN, ROW_B_MAX = 0, 4  # rank 1~5

FILES = list("abcdefgh")
RANKS = list("12345678")
_CELL_A = config.BOARD_MM_A / 8
_CELL_B = config.BOARD_MM_B / 8


# ─────────────────────────────────────────────────────────────
# 로봇 식별자
# ─────────────────────────────────────────────────────────────
class Robot(Enum):
    A = "A"
    B = "B"


# ─────────────────────────────────────────────────────────────
# 로봇별 좌표 변환
# ─────────────────────────────────────────────────────────────
def square_to_mm_for(robot: Robot, square: str) -> tuple:
    col = FILES.index(square[0].lower())
    row = RANKS.index(square[1])

    if robot == Robot.A:
        x = config.ORIGIN_X_MM + (7 - row) * (_CELL_A - 2.0)
        y = config.ORIGIN_Y_MM + col * (_CELL_A + 0.3)
    else:
        x = config.ORIGIN_B_X + row * _CELL_B
        y = config.ORIGIN_B_Y + (7 - col) * _CELL_B

    return (round(x, 3), round(y, 3))


# ─────────────────────────────────────────────────────────────
# 단일 로봇 상태
# ─────────────────────────────────────────────────────────────
@dataclass
class RobotState:
    robot: Robot
    port: str
    home_x: float
    home_y: float
    home_r: float = 0.0
    device: Optional[Dobot] = field(default=None, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


# ─────────────────────────────────────────────────────────────
# 비상 정지 플래그
# ─────────────────────────────────────────────────────────────
_abort_flag = False


def request_abort() -> None:
    global _abort_flag
    _abort_flag = True


def clear_abort() -> None:
    global _abort_flag
    _abort_flag = False


# ─────────────────────────────────────────────────────────────
# 메인 컨트롤러
# ─────────────────────────────────────────────────────────────
class DualDobotController:

    def __init__(self, log_fn: Callable = print):
        self.log = log_fn
        self._ra = RobotState(Robot.A, DOBOT_PORT_A, HOME_A_X, HOME_A_Y, HOME_A_R)
        self._rb = RobotState(Robot.B, DOBOT_PORT_B, HOME_B_X, HOME_B_Y, HOME_B_R)

    # ─────────────────────────────────────────────────────────
    # 초기화 / 종료
    # ─────────────────────────────────────────────────────────
    def init(self) -> None:
        """두 도봇 병렬 연결 → 내장 홈 → 대기점."""
        threads = [
            threading.Thread(target=self._init_robot, args=(self._ra,), daemon=True),
            threading.Thread(target=self._init_robot, args=(self._rb,), daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.log("[Dual] 두 로봇 초기화 완료", "ok")

    def _init_robot(self, rs: RobotState) -> None:
        self.log(f"[Robot {rs.robot.value}] 연결 중: {rs.port}", "info")
        rs.device = Dobot(port=rs.port, verbose=config.DOBOT_VERBOSE)
        self._clear_alarms(rs)
        self._lift_to_safe(rs)
        self._send_queue_reset(rs)
        self._do_home_cmd(rs)
        self._clear_alarms(rs)
        self._go_standby(rs)
        self.log(f"[Robot {rs.robot.value}] 준비 완료", "ok")

    def quit(self) -> None:
        for rs in (self._ra, self._rb):
            if rs.device:
                try:
                    rs.device.close()
                except Exception:
                    pass
                rs.device = None

    # ─────────────────────────────────────────────────────────
    # 홈 이동 (GUI 버튼용)
    # ─────────────────────────────────────────────────────────
    def go_home(self, robot: Optional[Robot] = None) -> None:
        """내장 홈 → 대기점. robot=None 이면 양쪽 동시."""

        def _home_and_standby(rs: RobotState) -> None:
            self._do_home_cmd(rs)
            self._go_standby(rs)

        if robot is not None:
            _home_and_standby(self._get_rs(robot))
            return
        threads = [
            threading.Thread(target=_home_and_standby, args=(self._ra,), daemon=True),
            threading.Thread(target=_home_and_standby, args=(self._rb,), daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # ─────────────────────────────────────────────────────────
    # 에러 초기화 (GUI 버튼용)
    # ─────────────────────────────────────────────────────────
    def recover_to_standby(self) -> None:
        """알람 클리어 후 대기점 복귀. 내장 홈 동작 없음."""
        for rs in (self._ra, self._rb):
            self._clear_alarms(rs)
        threads = [
            threading.Thread(target=self._go_standby, args=(self._ra,), daemon=True),
            threading.Thread(target=self._go_standby, args=(self._rb,), daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.log("[Dual] 에러 초기화 완료 — 대기점 복귀", "ok")

    # ─────────────────────────────────────────────────────────
    # 비상 정지
    # ─────────────────────────────────────────────────────────
    def emergency_stop_and_recover(self) -> None:
        """즉시 정지 → 그리퍼 off → 대기점 복귀."""
        request_abort()
        for rs in (self._ra, self._rb):
            if rs.device is None:
                continue
            try:
                self._send_queue_stop(rs)
                rs.device.grip(False)
                time.sleep(0.3)
            except Exception:
                pass
            self._clear_alarms(rs)
        threads = [
            threading.Thread(target=self._go_standby, args=(self._ra,), daemon=True),
            threading.Thread(target=self._go_standby, args=(self._rb,), daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        clear_abort()
        self.log("[Dual] 비상 정지 완료 — 대기점 복귀", "warn")

    # ─────────────────────────────────────────────────────────
    # 수 실행
    # ─────────────────────────────────────────────────────────
    def execute_move(
        self,
        board: chess.Board,
        move: chess.Move,
    ) -> None:
        uci = move.uci()
        from_sq = uci[:2]
        to_sq = uci[2:4]
        from_row = RANKS.index(from_sq[1])
        to_row = RANKS.index(to_sq[1])

        self.log(f"[Dual] 수 실행: {from_sq.upper()} → {to_sq.upper()}", "info")

        # ── ① 캡처: 상대 말 먼저 묘지로 ──────────────────────
        if board.is_capture(move):
            if board.is_en_passant(move):
                ep_rank = "5" if board.turn == chess.WHITE else "4"
                cap_sq = to_sq[0] + ep_rank
            else:
                cap_sq = to_sq

            cap_row = RANKS.index(cap_sq[1])
            cap_robot = Robot.A if ROW_A_MIN <= cap_row <= ROW_A_MAX else Robot.B
            cap_mm = square_to_mm_for(cap_robot, cap_sq)
            gy_mm = self._next_graveyard(board.turn)
            self.log(
                f"[Dual] 캡처: {cap_sq.upper()} → 묘지 (Robot {cap_robot.value})",
                "info",
            )
            self._do_single(cap_row, cap_mm, gy_mm, f"캡처({cap_sq.upper()})")

        # ── ② 말 이동 ──────────────────────────────────────────
        worker = self._select_robot(from_row, to_row)

        if worker is None:
            picker_robot = Robot.A if from_row >= ROW_A_MIN else Robot.B
            placer_robot = self._other(picker_robot)
            from_mm = square_to_mm_for(picker_robot, from_sq)
            to_mm = square_to_mm_for(placer_robot, to_sq)
            self.log(
                f"[Dual] Handoff: Robot {picker_robot.value} 픽 → Robot {placer_robot.value} 플레이스",
                "info",
            )
            self._handoff_move(from_mm, to_mm, picker_robot, placer_robot)
        else:
            from_mm = square_to_mm_for(worker, from_sq)
            to_mm = square_to_mm_for(worker, to_sq)
            self._park_other(self._other(worker))
            self._pick_and_place_rs(
                self._get_rs(worker),
                from_mm,
                to_mm,
                label=f"{from_sq.upper()}→{to_sq.upper()}",
            )

        self.log("[Dual] 수 실행 완료", "ok")

    # ─────────────────────────────────────────────────────────
    # 로봇 선택
    # ─────────────────────────────────────────────────────────
    def _select_robot(self, from_row: int, to_row: int) -> Optional[Robot]:
        a_ok = ROW_A_MIN <= from_row <= ROW_A_MAX and ROW_A_MIN <= to_row <= ROW_A_MAX
        b_ok = ROW_B_MIN <= from_row <= ROW_B_MAX and ROW_B_MIN <= to_row <= ROW_B_MAX

        if a_ok and b_ok:
            mid_row = (from_row + to_row) / 2
            mid_x = config.ORIGIN_X_MM + mid_row * _CELL_A
            return Robot.A if abs(mid_x - HOME_A_X) <= abs(mid_x - HOME_B_X) else Robot.B
        if a_ok:
            return Robot.A
        if b_ok:
            return Robot.B
        return None

    def _other(self, robot: Robot) -> Robot:
        return Robot.B if robot == Robot.A else Robot.A

    def _get_rs(self, robot: Robot) -> RobotState:
        return self._ra if robot == Robot.A else self._rb

    # ─────────────────────────────────────────────────────────
    # Handoff
    # ─────────────────────────────────────────────────────────
    def _handoff_move(self, from_mm, to_mm, picker_robot, placer_robot):
        # 각 로봇 좌표계 기준 handoff 위치
        handoff_picker = (
            (HANDOFF_A_X, HANDOFF_A_Y)
            if picker_robot == Robot.A
            else (HANDOFF_B_X, HANDOFF_B_Y)
        )
        handoff_placer = (
            (HANDOFF_A_X, HANDOFF_A_Y)
            if placer_robot == Robot.A
            else (HANDOFF_B_X, HANDOFF_B_Y)
        )

        picker = self._get_rs(picker_robot)
        placer = self._get_rs(placer_robot)

        self._park_other(placer_robot)
        self._pick_and_place_rs(picker, from_mm, handoff_picker, "HANDOFF-PICK")

        self._park_other(picker_robot)
        self._pick_and_place_rs(placer, handoff_placer, to_mm, "HANDOFF-PLACE")

    # ─────────────────────────────────────────────────────────
    # 단일 픽앤플레이스 (캡처용)
    # ─────────────────────────────────────────────────────────
    def _do_single(self, row: int, from_mm: tuple, to_mm: tuple, label: str) -> None:
        rs = self._ra if ROW_A_MIN <= row <= ROW_A_MAX else self._rb
        other = self._other(rs.robot)
        self._park_other(other)
        self._pick_and_place_rs(rs, from_mm, to_mm, label=label)

    # ─────────────────────────────────────────────────────────
    # 픽앤플레이스 (저수준)
    # ─────────────────────────────────────────────────────────
    def _pick_and_place_rs(
        self,
        rs: RobotState,
        from_mm: tuple,
        to_mm: tuple,
        label: str = "",
    ) -> None:
        if _abort_flag:
            raise RuntimeError("동작 중단 요청됨")

        self.log(
            f"[Robot {rs.robot.value}] {label}  "
            f"({from_mm[0]:.0f},{from_mm[1]:.0f}) → "
            f"({to_mm[0]:.0f},{to_mm[1]:.0f})",
            "info",
        )

        dev = rs.device
        zs = config.Z_SAFE_HEIGHT
        fx, fy = from_mm
        tx, ty = to_mm

        def mv(x, y, z):
            if _abort_flag:
                raise RuntimeError("동작 중단 요청됨")
            dev.move_to(x, y, z, config.Z_R, wait=True)
            time.sleep(config.MOVE_SETTLE)

        # 로봇별 Z 높이
        z_pick  = config.Z_PICK_A  if rs.robot == Robot.A else config.Z_PICK_B
        z_place = config.Z_PLACE_A if rs.robot == Robot.A else config.Z_PLACE_B

        # 1) 경유점
        self._go_waypoint(rs)

        # 2) PICK
        mv(fx, fy, zs)
        mv(fx, fy, z_pick)
        dev.grip(True)
        time.sleep(config.GRIP_WAIT)
        mv(fx, fy, zs)

        # 3) 경유점 (픽→플레이스 사이)
        self._go_waypoint(rs)

        # 4) PLACE
        mv(tx, ty, zs)
        mv(tx, ty, z_place)
        dev.grip(False)
        time.sleep(config.GRIP_WAIT)
        mv(tx, ty, zs)

        # 5) 경유점
        self._go_waypoint(rs)

        # 6) 대기점
        self._go_standby(rs)

        self.log(f"[Robot {rs.robot.value}] {label} 완료", "ok")

    # ─────────────────────────────────────────────────────────
    # 대기점 이동
    # ─────────────────────────────────────────────────────────
    def _park_other(self, robot: Robot) -> None:
        """비작업 로봇을 대기점으로 이동 (충돌 방지)."""
        self.log(f"[Robot {robot.value}] 대기점 이동", "info")
        self._go_standby(self._get_rs(robot))

    def _go_standby(self, rs: RobotState) -> None:
        """
        대기점으로 안전 이동.
        경유점 경유 후 대기점 XY로 이동.
        """
        if rs.device is None:
            return
        zs = config.Z_SAFE_HEIGHT
        try:
            pose = rs.device.pose()
            cx, cy, cz = pose[0], pose[1], pose[2]

            # 현재 J1 vs 대기점 J1 각도 차이 계산
            j1_now = math.degrees(math.atan2(cy, cx))
            j1_target = math.degrees(math.atan2(rs.home_y, rs.home_x))
            diff = abs(j1_target - j1_now)
            if diff > 180:
                diff = 360 - diff

            if diff >= config.WAYPOINT_J1_DIFF:
                # 각도 차이 클 때만 경유점 경유
                rs.device.move_to(WAYPOINT_X, WAYPOINT_Y, cz, WAYPOINT_R, wait=True)
                time.sleep(config.MOVE_SETTLE)
                if cz < zs - 1.0:
                    rs.device.move_to(WAYPOINT_X, WAYPOINT_Y, zs, WAYPOINT_R, wait=True)
                    time.sleep(config.MOVE_SETTLE)

        except Exception as e:
            self.log(f"[Robot {rs.robot.value}] 경유점 이동 오류: {e}", "err")

        try:
            rs.device.move_to(rs.home_x, rs.home_y, zs, rs.home_r, wait=True)
            time.sleep(config.MOVE_SETTLE)
        except Exception as e:
            self.log(f"[Robot {rs.robot.value}] 대기점 이동 오류: {e}", "err")

    # ─────────────────────────────────────────────────────────
    # 경유점 이동
    # ─────────────────────────────────────────────────────────
    def _go_waypoint(self, rs: RobotState) -> None:
        """
        경유점으로 이동.
        픽앤플레이스 시작 전/후에 들러 관절 안전 구간 확보.
        """
        if rs.device is None:
            return
        zs = config.Z_SAFE_HEIGHT
        try:
            pose = rs.device.pose()
            cz = pose[2]
            # 현재 Z 먼저 안전고도로 올림
            if cz < zs - 1.0:
                rs.device.move_to(pose[0], pose[1], zs, pose[3], wait=True)
                time.sleep(config.MOVE_SETTLE)
            # 경유점으로 이동
            rs.device.move_to(WAYPOINT_X, WAYPOINT_Y, zs, WAYPOINT_R, wait=True)
            time.sleep(config.MOVE_SETTLE)
        except Exception as e:
            self.log(f"[Robot {rs.robot.value}] 경유점 이동 오류: {e}", "err")

    # ─────────────────────────────────────────────────────────
    # 묘지
    # ─────────────────────────────────────────────────────────
    def _next_graveyard(self, attacker_color: chess.Color) -> tuple:
        if attacker_color == chess.WHITE:
            return GRAVEYARD_B
        else:
            return GRAVEYARD_A

    # ─────────────────────────────────────────────────────────
    # 진단
    # ─────────────────────────────────────────────────────────
    def get_pose(self, robot: Robot) -> tuple:
        rs = self._get_rs(robot)
        if rs.device is None:
            raise RuntimeError(f"Robot {robot.value} 미연결")
        p = rs.device.pose()
        return p[0], p[1], p[2], p[3]

    def get_status(self) -> dict:
        out = {}
        for rs in (self._ra, self._rb):
            if rs.device:
                try:
                    p = rs.device.pose()
                    out[rs.robot.value] = {
                        "connected": True,
                        "x": round(p[0], 1),
                        "y": round(p[1], 1),
                        "z": round(p[2], 1),
                    }
                except Exception as e:
                    out[rs.robot.value] = {"connected": False, "error": str(e)}
            else:
                out[rs.robot.value] = {"connected": False}
        return out

    # ─────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────────────────────
    def _clear_alarms(self, rs: RobotState) -> None:
        if rs.device is None:
            return
        for _ in range(3):
            msg = Message()
            msg.id = CommunicationProtocolIDs.CLEAR_ALL_ALARMS_STATE
            msg.ctrl = ControlValues.ONE
            rs.device._send_command(msg)
            time.sleep(0.2)

    def _lift_to_safe(self, rs: RobotState) -> None:
        """초기화 시 위험 자세 탈출. 경유점 경유 후 대기점으로."""
        if rs.device is None:
            return
        try:
            pose = rs.device.pose()
            cx, cy, cz, cr = pose[0], pose[1], pose[2], pose[3]

            # 경유점 XY로 먼저 수평 이동 (현재 Z 유지)
            rs.device.move_to(WAYPOINT_X, WAYPOINT_Y, cz, WAYPOINT_R, wait=True)
            time.sleep(config.MOVE_SETTLE)

            # 경유점에서 안전고도로 상승
            rs.device.move_to(
                WAYPOINT_X, WAYPOINT_Y, config.Z_SAFE_HEIGHT, WAYPOINT_R, wait=True
            )
            time.sleep(config.MOVE_SETTLE)

        except Exception as e:
            self.log(f"[Robot {rs.robot.value}] lift_to_safe 오류: {e}", "warn")

    def _do_home_cmd(self, rs: RobotState) -> None:
        """도봇 내장 홈 명령 (관절 캘리브레이션)."""
        if rs.device is None:
            return
        self.log(f"[Robot {rs.robot.value}] 홈 이동 중...", "info")
        msg = Message()
        msg.id = CommunicationProtocolIDs.SET_HOME_CMD
        msg.ctrl = ControlValues.THREE
        msg.params = bytearray([0x00])
        resp = rs.device._send_command(msg)
        if resp and resp.params:
            idx = struct.unpack_from("L", resp.params, 0)[0]
            self._wait_for_cmd(rs, idx, timeout=30)
        else:
            time.sleep(5)
        self.log(f"[Robot {rs.robot.value}] 홈 완료", "ok")

    def _send_queue_reset(self, rs: RobotState) -> None:
        if rs.device is None:
            return
        for cmd_id in [
            CommunicationProtocolIDs.SET_QUEUED_CMD_STOP_EXEC,
            CommunicationProtocolIDs.SET_QUEUED_CMD_CLEAR,
            CommunicationProtocolIDs.SET_QUEUED_CMD_START_EXEC,
        ]:
            msg = Message()
            msg.id = cmd_id
            msg.ctrl = ControlValues.ONE
            rs.device._send_command(msg)
            time.sleep(0.2)

    def _send_queue_stop(self, rs: RobotState) -> None:
        if rs.device is None:
            return
        for cmd_id in [
            CommunicationProtocolIDs.SET_QUEUED_CMD_STOP_EXEC,
            CommunicationProtocolIDs.SET_QUEUED_CMD_CLEAR,
            CommunicationProtocolIDs.SET_QUEUED_CMD_START_EXEC,
        ]:
            msg = Message()
            msg.id = cmd_id
            msg.ctrl = ControlValues.ONE
            try:
                rs.device._send_command(msg)
            except Exception:
                pass
            time.sleep(0.2)

    def _wait_for_cmd(
        self, rs: RobotState, expected_idx: int, timeout: float = 30
    ) -> None:
        start = time.time()
        while True:
            msg = Message()
            msg.id = CommunicationProtocolIDs.GET_QUEUED_CMD_CURRENT_INDEX
            resp = rs.device._send_command(msg)
            if resp and resp.params:
                cur = struct.unpack_from("L", resp.params, 0)[0]
                if cur >= expected_idx:
                    break
            if time.time() - start > timeout:
                print(f"[Robot {rs.robot.value}] 홈 이동 타임아웃")
                break
            time.sleep(0.1)