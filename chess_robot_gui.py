"""
chess_robot_gui.py
------------------
체스 로봇 통합 GUI — 듀얼 도봇 버전.

실행:
    QT_QPA_PLATFORM=xcb python chess_robot_gui.py

의존:
    pip install pillow python-chess torch torchvision
    sudo apt install stockfish

변경사항:
  - 모델 로드 후 _try_upgrade_to_vision_ctrl() 자동 호출
  - 재캘리브 완료 후 _try_upgrade_to_vision_ctrl() 자동 호출
  - _try_upgrade_to_vision_ctrl(): 모델+캘리브 둘 다 있으면 VisionGuidedController로 자동 전환
  - _load_model(): backbone. 키 접두사 자동 보정 추가, weights_only=False로 변경
  - _manual_move(): 비전 컨트롤러일 때도 execute_move 경유하도록 통일
  - _execute_castling_rook(): 비전 컨트롤러일 때 vision pick/place 사용하도록 수정
"""

import threading
import time
import tkinter as tk
from tkinter import messagebox

import chess
import cv2
import numpy as np
from PIL import Image, ImageTk

import config
from dual_dobot_controller import (
    DualDobotController, Robot,
    request_abort, square_to_mm_for,
    ROW_A_MIN, ROW_A_MAX, RANKS,
)
from vision_coord import VisionRobotCalib
from vision_coord_patch import VisionGuidedController
from chess_vision import ChessVision
from chess_engine import ChessEngine, board_from_vision


# ─────────────────────────────────────────────────────────────
# 테마 상수
# ─────────────────────────────────────────────────────────────
BG    = "#1e1e1e"
CARD  = "#2a2a2a"
ACC   = "#4a9eff"
FG    = "#e0e0e0"
MUT   = "#888888"
FONT  = ("Consolas", 10)
HFONT = ("Consolas", 11, "bold")
SFONT = ("Consolas", 9)

COLOR_OK   = "#7fff9a"
COLOR_WARN = "#ffd080"
COLOR_ERR  = "#ff7f7f"
COLOR_INFO = "#7fbfff"


# ─────────────────────────────────────────────────────────────
# 메인 GUI 클래스
# ─────────────────────────────────────────────────────────────
class ChessRobotGUI:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Chess Robot Control — Dual Dobot")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.board              = chess.Board()
        self.pending_move: chess.Move | None = None
        self.vision:  ChessVision | None     = None
        self.engine:  ChessEngine | None     = None
        self.dual_ctrl: DualDobotController | None = None
        self.robot_busy  = False
        self.cam_running = False

        self._build_ui()
        self._init_systems()
        self._start_camera_thread()

    # ─────────────────────────────────────────────────────────
    # UI 빌드
    # ─────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg=BG, pady=4)
        top.pack(fill="x", padx=12)

        badge_row = tk.Frame(top, bg=BG)
        badge_row.pack(fill="x")
        self._badge_robot_a = self._make_badge(badge_row, "로봇A", "연결 중...", COLOR_WARN)
        self._badge_robot_b = self._make_badge(badge_row, "로봇B", "연결 중...", COLOR_WARN)
        self._badge_vision  = self._make_badge(badge_row, "비전",  "초기화 중...", COLOR_WARN)
        self._badge_sf      = self._make_badge(badge_row, "Stockfish", "로드 중...", COLOR_WARN)
        self._badge_coord   = self._make_badge(badge_row, "좌표", "수동", COLOR_WARN)
        for b in (self._badge_robot_a, self._badge_robot_b,
                  self._badge_vision, self._badge_sf, self._badge_coord):
            b.pack(side="left", padx=(0, 8))

        # ── 1번째 버튼 행
        btn_row1 = tk.Frame(top, bg=BG, pady=2)
        btn_row1.pack(fill="x")
        for text, cmd, bg_, fg_ in [
            ("🔲 재캘리브",      self._recalib,            "#2a2a3a", "#aaaaff"),
            ("📷 보드 스캔",     self._scan_board,          "#333",    FG),
            ("♟ 초기 보드 세팅", self._reset_board,         "#1a3a5c", "#aaddff"),
            ("♟ SF 재연결",      self._reconnect_stockfish, "#1a3a1a", "#aaffaa"),
        ]:
            tk.Button(btn_row1, text=text, bg=bg_, fg=fg_, font=SFONT,
                      relief="flat", cursor="hand2", command=cmd,
                      ).pack(side="left", padx=(0, 6), ipady=3, ipadx=6)

        tk.Button(btn_row1, text="🧠 모델 불러오기",
                  bg="#2a3a2a", fg="#aaffaa", font=SFONT,
                  relief="flat", cursor="hand2", command=self._load_model,
                  ).pack(side="left", ipady=3, ipadx=6)
        self.model_path_var = tk.StringVar(value="더미 모드")
        tk.Label(btn_row1, textvariable=self.model_path_var,
                 bg=BG, fg=MUT, font=("Consolas", 8),
                 wraplength=300, justify="left").pack(side="left", padx=(6, 0))

        # ── 2번째 버튼 행
        btn_row2 = tk.Frame(top, bg=BG, pady=2)
        btn_row2.pack(fill="x")
        for text, cmd, bg_, fg_ in [
            ("⚠ 에러 초기화",  self._clear_alarm,                "#3a1a1a", "#ff9999"),
            ("🏠 홈 이동",     self._go_home,                    "#2a3a2a", "#aaffaa"),
            ("⏹ 즉시 정지",    self._emergency_stop,             "#5c1a1a", "#ff4444"),
            ("📷 비전 재연결",  self._reconnect_vision,           "#2a2a3a", "#aaaaff"),
            ("🔌 로봇A 재연결", lambda: self._reconnect(Robot.A), "#1a2a3a", "#aaccff"),
            ("🔌 로봇B 재연결", lambda: self._reconnect(Robot.B), "#1a2a3a", "#aaccff"),
        ]:
            tk.Button(btn_row2, text=text, bg=bg_, fg=fg_, font=SFONT,
                      relief="flat", cursor="hand2", command=cmd,
                      ).pack(side="left", padx=(0, 6), ipady=3, ipadx=6)

        tk.Button(btn_row2, text="📸 데이터 캡처",
                  bg="#3a2a1a", fg="#ffcc88", font=SFONT,
                  relief="flat", cursor="hand2", command=self._capture_dataset,
                  ).pack(side="left", padx=(0, 6), ipady=3, ipadx=6)

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=4)
        main.columnconfigure(0, weight=2)
        main.columnconfigure(1, weight=3)
        main.rowconfigure(0, weight=1)

        left  = tk.Frame(main, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right = tk.Frame(main, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent: tk.Frame) -> None:
        bc = self._card(parent, "BOARD")
        self.board_canvas = tk.Canvas(bc, width=256, height=256,
                                      bg=CARD, highlightthickness=0)
        self.board_canvas.pack(padx=10, pady=(0, 10))
        self._draw_board()

        sf = self._card(parent, "STOCKFISH 추천")
        self.sf_move_var = tk.StringVar(value="—")
        self.sf_eval_var = tk.StringVar(value="")
        tk.Label(sf, textvariable=self.sf_move_var,
                 bg=CARD, fg=ACC, font=("Consolas", 20, "bold")).pack(padx=10)
        tk.Label(sf, textvariable=self.sf_eval_var,
                 bg=CARD, fg=MUT, font=FONT).pack(padx=10, pady=(0, 4))

        btn_row = tk.Frame(sf, bg=CARD)
        btn_row.pack(fill="x", padx=10, pady=(4, 10))
        self.btn_approve = tk.Button(
            btn_row, text="✓  승인 — 도봇 실행",
            bg="#1a5c2a", fg=COLOR_OK, font=HFONT,
            relief="flat", cursor="hand2",
            command=self._approve_move, state="disabled",
        )
        self.btn_approve.pack(side="left", fill="x", expand=True, padx=(0, 4), ipady=6)
        self.btn_reject = tk.Button(
            btn_row, text="✗  거절",
            bg="#5c1a1a", fg=COLOR_ERR, font=HFONT,
            relief="flat", cursor="hand2",
            command=self._reject_move, state="disabled",
        )
        self.btn_reject.pack(side="left", ipady=6, ipadx=10)

        mc   = self._card(parent, "수동 이동")
        mrow = tk.Frame(mc, bg=CARD)
        mrow.pack(fill="x", padx=10, pady=(0, 6))
        kw = dict(width=4, font=("Consolas", 14, "bold"), justify="center",
                  bg="#333", fg=FG, insertbackground=FG, relief="flat")
        tk.Button(mrow, text="♟ SF 분석", bg="#1a3a1a", fg=COLOR_OK, font=HFONT,
                  relief="flat", cursor="hand2",
                  command=self._analyze).pack(side="left", padx=(0, 12), ipadx=8, ipady=4)
        self.entry_from = tk.Entry(mrow, **kw)
        self.entry_from.pack(side="left")
        tk.Label(mrow, text="→", bg=CARD, fg=MUT, font=HFONT).pack(side="left", padx=6)
        self.entry_to = tk.Entry(mrow, **kw)
        self.entry_to.pack(side="left")
        tk.Button(mrow, text="실행", bg="#333", fg=ACC, font=HFONT,
                  relief="flat", cursor="hand2",
                  command=self._manual_move).pack(side="left", padx=(8, 0), ipadx=8)
        self.entry_from.bind("<Return>", lambda e: self.entry_to.focus())
        self.entry_to.bind("<Return>",   lambda e: self._manual_move())

        grip_row = tk.Frame(mc, bg=CARD)
        grip_row.pack(fill="x", padx=10, pady=(0, 10))
        for text, robot, val in [
            ("A 잡기", Robot.A, True),
            ("A 놓기", Robot.A, False),
            ("B 잡기", Robot.B, True),
            ("B 놓기", Robot.B, False),
        ]:
            tk.Button(grip_row, text=text, bg="#2a2a2a", fg=COLOR_WARN, font=SFONT,
                      relief="flat", cursor="hand2",
                      command=lambda r=robot, v=val: self._test_grip(r, v),
                      ).pack(side="left", padx=(0, 6), ipady=3, ipadx=6)

    def _build_right(self, parent: tk.Frame) -> None:
        cam_card = tk.Frame(parent, bg=CARD)
        cam_card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        tk.Label(cam_card, text="LIVE CAMERA", bg=CARD, fg=MUT,
                 font=SFONT, anchor="w").pack(fill="x", padx=10, pady=(8, 2))
        self.cam_label = tk.Label(cam_card, bg="#111",
                                  text="카메라 초기화 중...", fg=MUT, font=FONT)
        self.cam_label.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        log_card = tk.Frame(parent, bg=CARD)
        log_card.grid(row=1, column=0, sticky="nsew")
        tk.Label(log_card, text="LOG", bg=CARD, fg=MUT,
                 font=SFONT, anchor="w").pack(fill="x", padx=10, pady=(8, 2))
        self.log_text = tk.Text(
            log_card, height=6, bg="#111", fg="#aaa",
            font=("Consolas", 9), relief="flat", state="disabled", wrap="word",
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        for tag, color in [("ok", COLOR_OK), ("warn", COLOR_WARN),
                           ("err", COLOR_ERR), ("info", COLOR_INFO)]:
            self.log_text.tag_config(tag, foreground=color)

    # ─────────────────────────────────────────────────────────
    # UI 헬퍼
    # ─────────────────────────────────────────────────────────
    def _card(self, parent, title):
        f = tk.Frame(parent, bg=CARD)
        f.pack(fill="x", pady=(0, 8))
        tk.Label(f, text=title, bg=CARD, fg=MUT,
                 font=SFONT, anchor="w").pack(fill="x", padx=10, pady=(8, 4))
        return f

    def _make_badge(self, parent, label, text, color):
        f = tk.Frame(parent, bg=CARD, padx=8, pady=3)
        tk.Label(f, text=label, bg=CARD, fg="#666", font=SFONT).pack(side="left", padx=(0, 4))
        lbl = tk.Label(f, text=text, bg=CARD, fg=color, font=("Consolas", 9, "bold"))
        lbl.pack(side="left")
        f._label = lbl
        return f

    def _set_badge(self, badge, text, color):
        badge._label.config(text=text, fg=color)

    def log(self, msg: str, tag: str = "") -> None:
        def _do():
            ts = time.strftime("%H:%M:%S")
            self.log_text.config(state="normal")
            self.log_text.insert("end", f"[{ts}] {msg}\n", tag)
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.root.after(0, _do)

    # ─────────────────────────────────────────────────────────
    # 체스보드 그리기
    # ─────────────────────────────────────────────────────────
    def _draw_board(self, highlight_from=None, highlight_to=None) -> None:
        c  = self.board_canvas
        c.delete("all")
        SZ = 32
        light, dark    = "#f0d9b5", "#b58863"
        hl_from, hl_to = "#7fc97f", "#4caf50"

        for r in range(8):
            for f in range(8):
                sq = chess.square(r, f)
                x0, y0 = (7 - f) * SZ, (7 - r) * SZ
                if highlight_from is not None and sq == highlight_from:
                    fill = hl_from
                elif highlight_to is not None and sq == highlight_to:
                    fill = hl_to
                else:
                    fill = light if (r + f) % 2 != 0 else dark
                c.create_rectangle(x0, y0, x0 + SZ, y0 + SZ, fill=fill, outline="")
                piece = self.board.piece_at(sq)
                if piece:
                    sym = piece.unicode_symbol()
                    fg  = "#fff" if piece.color == chess.WHITE else "#111"
                    c.create_text(x0 + SZ // 2, y0 + SZ // 2,
                                  text=sym, font=("Arial", 18), fill=fg)

        for i, f in enumerate("abcdefgh"):
            c.create_text(i * SZ + SZ // 2, 256 - 6, text=f,
                          font=("Consolas", 7), fill="#888")
        for i in range(8):
            c.create_text(4, i * SZ + SZ // 2, text=str(8 - i),
                          font=("Consolas", 7), fill="#888")

    # ─────────────────────────────────────────────────────────
    # 시스템 초기화
    # ─────────────────────────────────────────────────────────
    def _init_systems(self) -> None:
        def _run():
            # 비전
            try:
                self.log("카메라 초기화 중...", "info")
                self.vision = ChessVision()
                status = "더미모드" if self.vision.dummy_mode else "모델OK"
                color  = COLOR_WARN if self.vision.dummy_mode else COLOR_OK
                self.root.after(0, lambda: self._set_badge(self._badge_vision, status, color))
                self.log(f"비전 초기화 완료 ({status})", "ok")
                if self.vision.H is None:
                    self.log("캘리브레이션 없음 → 재캘리브 버튼 눌러주세요", "warn")
            except Exception as e:
                self.log(f"비전 오류: {e}", "err")
                self.root.after(0, lambda: self._set_badge(self._badge_vision, "오류", COLOR_ERR))

            # 듀얼 도봇
            try:
                self.log("도봇 A/B 초기화 중...", "info")
                calib = VisionRobotCalib()
                if self.vision and not self.vision.dummy_mode and calib.is_ready:
                    self.dual_ctrl = VisionGuidedController(
                        vision=self.vision,
                        calib=calib,
                        log_fn=self.log,
                    )
                    self.log("비전 좌표 모드로 초기화", "ok")
                    self.root.after(0, lambda: self._set_badge(self._badge_coord, "비전", COLOR_OK))
                else:
                    self.dual_ctrl = DualDobotController(log_fn=self.log)
                    self.log("수동 좌표 모드로 초기화", "warn")
                    self.root.after(0, lambda: self._set_badge(self._badge_coord, "수동", COLOR_WARN))
                self.dual_ctrl.init()
                self.root.after(0, lambda: self._set_badge(self._badge_robot_a, "연결됨", COLOR_OK))
                self.root.after(0, lambda: self._set_badge(self._badge_robot_b, "연결됨", COLOR_OK))
            except Exception as e:
                self.log(f"도봇 오류: {e}", "err")
                self.root.after(0, lambda: self._set_badge(self._badge_robot_a, "오류", COLOR_ERR))
                self.root.after(0, lambda: self._set_badge(self._badge_robot_b, "오류", COLOR_ERR))

            # Stockfish
            try:
                self.log("Stockfish 로드 중...", "info")
                self.engine = ChessEngine()
                self.root.after(0, lambda: self._set_badge(self._badge_sf, "준비됨", COLOR_OK))
                self.log("Stockfish 연결 완료", "ok")
            except Exception as e:
                self.log(f"Stockfish 오류: {e}", "err")
                self.root.after(0, lambda: self._set_badge(self._badge_sf, "없음", COLOR_ERR))

        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # [핵심] 비전 좌표 모드 자동 전환
    # 모델 로드 or 캘리브 완료 시 자동 호출
    # 모델 + 캘리브 둘 다 있으면 VisionGuidedController로 전환
    # ─────────────────────────────────────────────────────────
    def _try_upgrade_to_vision_ctrl(self) -> None:
        if self.vision is None or self.vision.dummy_mode:
            self.log("비전 모드 전환 불가 — 모델 먼저 로드하세요", "warn")
            return
        if self.dual_ctrl is None:
            return

        calib = VisionRobotCalib()
        if not calib.is_ready:
            self.log("비전 좌표 대기 중 — 재캘리브 버튼으로 캘리브레이션 후 자동 전환됩니다", "warn")
            return

        if isinstance(self.dual_ctrl, VisionGuidedController):
            # 이미 비전 모드 → calib만 업데이트
            self.dual_ctrl.calib = calib
            self.log("비전 캘리브 업데이트 완료", "ok")
        else:
            # 수동 모드 → 비전 모드로 교체 (도봇 연결은 유지)
            old_ctrl = self.dual_ctrl
            self.dual_ctrl = VisionGuidedController(
                vision=self.vision,
                calib=calib,
                log_fn=self.log,
            )
            # 기존 도봇 연결 상태 이전
            self.dual_ctrl._ra = old_ctrl._ra
            self.dual_ctrl._rb = old_ctrl._rb
            self.log("비전 좌표 모드로 전환 완료", "ok")

        self.root.after(0, lambda: self._set_badge(self._badge_coord, "비전", COLOR_OK))

    # ─────────────────────────────────────────────────────────
    # 카메라 스레드
    # ─────────────────────────────────────────────────────────
    def _start_camera_thread(self) -> None:
        self.cam_running = True
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self) -> None:
        interval = 1.0 / config.CAMERA_FPS
        while self.cam_running:
            if self.vision is None:
                time.sleep(0.1)
                continue
            try:
                frame = self.vision._grab()
                if self.vision.H is not None:
                    display = self.vision._warp(frame)
                    self._draw_grid_overlay(display)
                    if self.pending_move:
                        self._highlight_move_overlay(display, self.pending_move)
                else:
                    display = frame.copy()
                    cv2.putText(display, "'재캘리브' 버튼을 눌러 캘리브레이션하세요",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

                rgb    = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                img    = Image.fromarray(rgb)
                lw, lh = self.cam_label.winfo_width(), self.cam_label.winfo_height()
                if lw > 10 and lh > 10:
                    img = img.resize((lw, lh), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.root.after(0, self._update_cam, imgtk)
            except Exception:
                pass
            time.sleep(interval)

    def _draw_grid_overlay(self, display) -> None:
        h, w = display.shape[:2]
        cw, ch = w // 8, h // 8
        for i in range(9):
            cv2.line(display, (i * cw, 0), (i * cw, h), (80, 80, 80), 1)
            cv2.line(display, (0, i * ch), (w, i * ch), (80, 80, 80), 1)

    def _highlight_move_overlay(self, display, move: chess.Move) -> None:
        h, w   = display.shape[:2]
        cw, ch = w // 8, h // 8
        fc, fr = 7 - move.from_square // 8, 7 - move.from_square % 8
        tc, tr = 7 - move.to_square // 8,   7 - move.to_square % 8
        cv2.rectangle(display, (fc*cw+1, fr*ch+1), ((fc+1)*cw-1, (fr+1)*ch-1), (100, 200, 100), 2)
        cv2.rectangle(display, (tc*cw+1, tr*ch+1), ((tc+1)*cw-1, (tr+1)*ch-1), (50, 150, 255), 2)

    def _update_cam(self, imgtk) -> None:
        self.cam_label.imgtk = imgtk
        self.cam_label.config(image=imgtk, text="")

    # ─────────────────────────────────────────────────────────
    # 보드 스캔
    # ─────────────────────────────────────────────────────────
    def _scan_board(self) -> None:
        if self.vision is None:
            self.log("비전 미초기화", "warn")
            return
        def _run():
            self.log("보드 스캔 중...", "info")
            try:
                piece_map  = self.vision.get_board()
                self.board = board_from_vision(piece_map)
                self.root.after(0, self._draw_board)
                self.log(f"스캔 완료 — {len(piece_map)}개 말 인식", "ok")
            except Exception as e:
                self.log(f"스캔 오류: {e}", "err")
        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # Stockfish 분석
    # ─────────────────────────────────────────────────────────
    def _analyze(self) -> None:
        if self.engine is None:
            self.log("Stockfish 없음", "err")
            return
        def _run():
            self.log("Stockfish 분석 중...", "info")
            self.root.after(0, lambda: self._set_badge(self._badge_sf, "분석 중...", COLOR_WARN))
            try:
                result = self.engine.analyse(self.board)
                self.pending_move = result.move
                self.root.after(0, lambda: self.sf_move_var.set(result.move_str))
                self.root.after(0, lambda: self.sf_eval_var.set(f"평가: {result.eval_str}"))
                self.root.after(0, lambda: self.btn_approve.config(state="normal"))
                self.root.after(0, lambda: self.btn_reject.config(state="normal"))
                self.root.after(0, lambda: self._draw_board(result.move.from_square, result.move.to_square))
                self.root.after(0, lambda: self._set_badge(self._badge_sf, "준비됨", COLOR_OK))
                self.log(f"추천: {result.move_str}  평가: {result.eval_str}", "ok")
            except Exception as e:
                self.log(f"분석 오류: {e}", "err")
                self.root.after(0, lambda: self._set_badge(self._badge_sf, "오류", COLOR_ERR))
        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # 승인
    # ─────────────────────────────────────────────────────────
    def _approve_move(self) -> None:
        if self.pending_move is None or self.robot_busy:
            return
        if self.dual_ctrl is None:
            self.log("도봇 미연결", "err")
            return

        move = self.pending_move
        self.pending_move = None
        self.btn_approve.config(state="disabled")
        self.btn_reject.config(state="disabled")
        self.sf_move_var.set("실행 중...")

        def _run():
            self.robot_busy = True
            try:
                is_castling = self.board.is_castling(move)
                self.dual_ctrl.execute_move(board=self.board, move=move)
                if is_castling:
                    self._execute_castling_rook(move)
                self.board.push(move)

                if self.board.is_checkmate():
                    winner = "백" if self.board.turn == chess.BLACK else "흑"
                    self.log(f"♚ 체크메이트! {winner} 승리", "ok")
                    self.root.after(0, lambda: messagebox.showinfo(
                        "게임 종료", f"체크메이트!\n{winner} 승리"))
                elif self.board.is_stalemate():
                    self.log("스테일메이트 — 무승부", "warn")
                    self.root.after(0, lambda: messagebox.showinfo("게임 종료", "스테일메이트\n무승부"))
                elif self.board.is_check():
                    self.log("⚠ 체크!", "warn")

                self.root.after(0, self._draw_board)
                self.root.after(0, lambda: self.sf_move_var.set("—"))
                self.root.after(0, lambda: self.sf_eval_var.set(""))
                self.log("이동 완료 — 다음 턴", "ok")
            except Exception as e:
                self.log(f"실행 오류: {e}", "err")
            finally:
                self.robot_busy = False

        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # 캐슬링 룩 물리 이동
    # 비전 모드면 vision pick/place 사용, 아니면 수동 좌표
    # ─────────────────────────────────────────────────────────
    def _execute_castling_rook(self, move: chess.Move) -> None:
        # board.push(move) 이전에 호출되므로 board.turn = 현재 수를 두는 색
        mover_is_white = self.board.turn == chess.WHITE
        if mover_is_white:
            rook_from, rook_to = ("h1", "f1") if move.to_square == chess.G1 else ("a1", "d1")
            rook_label = "wR"
        else:
            rook_from, rook_to = ("h8", "f8") if move.to_square == chess.G8 else ("a8", "d8")
            rook_label = "bR"

        from_row = RANKS.index(rook_from[1])
        to_row   = RANKS.index(rook_to[1])
        worker   = self.dual_ctrl._select_robot(from_row, to_row)

        if isinstance(self.dual_ctrl, VisionGuidedController):
            # 비전 모드: vision pick/place 사용
            robot = worker if worker else (Robot.A if from_row >= ROW_A_MIN else Robot.B)
            from_mm = self.dual_ctrl._vision_pick_mm(rook_from, robot, rook_label)
            to_mm   = self.dual_ctrl._vision_place_mm(rook_to, robot)
        else:
            # 수동 모드
            robot = worker if worker else (Robot.A if from_row >= ROW_A_MIN else Robot.B)
            from_mm = square_to_mm_for(robot, rook_from)
            to_mm   = square_to_mm_for(robot, rook_to)

        if worker is None:
            pk = Robot.A if from_row >= ROW_A_MIN else Robot.B
            pl = Robot.B if pk == Robot.A else Robot.A
            self.dual_ctrl._handoff_move(from_mm, to_mm, pk, pl)
        else:
            self.dual_ctrl._park_other(self.dual_ctrl._other(worker))
            self.dual_ctrl._pick_and_place_rs(
                self.dual_ctrl._get_rs(worker), from_mm, to_mm,
                label=f"캐슬링 룩({rook_from.upper()}→{rook_to.upper()})"
            )
            self.dual_ctrl._go_standby(self.dual_ctrl._get_rs(worker))
        self.log(f"캐슬링 룩: {rook_from.upper()} → {rook_to.upper()}", "info")

    def _reject_move(self) -> None:
        self.pending_move = None
        self.btn_approve.config(state="disabled")
        self.btn_reject.config(state="disabled")
        self.sf_move_var.set("—")
        self.sf_eval_var.set("")
        self._draw_board()
        self.log("추천 수 거절됨", "warn")

    # ─────────────────────────────────────────────────────────
    # 그리퍼 테스트
    # ─────────────────────────────────────────────────────────
    def _test_grip(self, robot: Robot, val: bool) -> None:
        if self.dual_ctrl is None:
            self.log("도봇 미연결", "err")
            return
        def _run():
            try:
                rs = self.dual_ctrl._get_rs(robot)
                if rs.device is None:
                    self.log(f"[Robot {robot.value}] 미연결", "err")
                    return
                rs.device.grip(val)
                self.log(f"[Robot {robot.value}] 그리퍼 {'잡기' if val else '놓기'}", "info")
            except Exception as e:
                self.log(f"[Robot {robot.value}] 그리퍼 오류: {e}", "err")
        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # 수동 이동 — execute_move 경유로 비전/수동 자동 분기
    # ─────────────────────────────────────────────────────────
    def _manual_move(self) -> None:
        from_sq = self.entry_from.get().strip().lower()
        to_sq   = self.entry_to.get().strip().lower()
        if not from_sq or not to_sq:
            self.log("출발/도착 칸 입력 필요", "warn")
            return
        if len(from_sq) < 2 or len(to_sq) < 2:
            self.log("칸 표기 오류 (예: e2, d4)", "warn")
            return
        if from_sq[0] not in "abcdefgh" or from_sq[1] not in "12345678":
            self.log(f"출발 칸 오류: {from_sq}", "warn")
            return
        if to_sq[0] not in "abcdefgh" or to_sq[1] not in "12345678":
            self.log(f"도착 칸 오류: {to_sq}", "warn")
            return
        if self.robot_busy:
            self.log("도봇 사용 중", "warn")
            return
        if self.dual_ctrl is None:
            self.log("도봇 미연결", "err")
            return

        def _run():
            self.robot_busy = True
            try:
                from_row = RANKS.index(from_sq[1])
                to_row   = RANKS.index(to_sq[1])
                try:
                    move = chess.Move.from_uci(from_sq + to_sq)
                except Exception:
                    move = None

                if move and move in self.board.legal_moves:
                    # 합법적인 수 → execute_move 경유 (비전/수동 자동 분기)
                    self.dual_ctrl.execute_move(self.board, move)
                    self.board.push(move)
                else:
                    # 비합법 수 or 좌표 직접 이동 → 수동 좌표 강제
                    worker = self.dual_ctrl._select_robot(from_row, to_row)
                    if worker is None:
                        pk = Robot.A if from_row >= ROW_A_MIN else Robot.B
                        pl = Robot.B if pk == Robot.A else Robot.A
                        from_mm = square_to_mm_for(pk, from_sq)
                        to_mm   = square_to_mm_for(pl, to_sq)
                        self.dual_ctrl._handoff_move(from_mm, to_mm, pk, pl)
                    else:
                        from_mm = square_to_mm_for(worker, from_sq)
                        to_mm   = square_to_mm_for(worker, to_sq)
                        self.dual_ctrl._park_other(self.dual_ctrl._other(worker))
                        self.dual_ctrl._pick_and_place_rs(
                            self.dual_ctrl._get_rs(worker), from_mm, to_mm,
                            label=f"수동({from_sq.upper()}→{to_sq.upper()})"
                        )
                        self.dual_ctrl._go_standby(self.dual_ctrl._get_rs(worker))

                self.root.after(0, self._draw_board)
                self.log(f"수동 이동 완료: {from_sq.upper()} → {to_sq.upper()}", "ok")
            except Exception as e:
                self.log(f"수동 이동 오류: {e}", "err")
            finally:
                self.robot_busy = False

        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # 홈 이동
    # ─────────────────────────────────────────────────────────
    def _go_home(self) -> None:
        if self.robot_busy:
            self.log("도봇 사용 중", "warn")
            return
        if self.dual_ctrl is None:
            self.log("도봇 미연결", "err")
            return
        def _run():
            self.robot_busy = True
            try:
                self.dual_ctrl.go_home()
                self.log("홈 이동 완료 (A/B)", "ok")
            except Exception as e:
                self.log(f"홈 이동 오류: {e}", "err")
            finally:
                self.robot_busy = False
        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # 즉시 정지
    # ─────────────────────────────────────────────────────────
    def _emergency_stop(self) -> None:
        request_abort()
        self.log("⏹ 즉시 정지 요청", "warn")
        def _run():
            try:
                if self.dual_ctrl:
                    self.dual_ctrl.emergency_stop_and_recover()
                self.log("비상 정지 완료", "ok")
            except Exception as e:
                self.log(f"정지 오류: {e}", "err")
            finally:
                self.robot_busy = False
        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # 알람 클리어
    # ─────────────────────────────────────────────────────────
    def _clear_alarm(self) -> None:
        if self.robot_busy:
            self.log("도봇 사용 중", "warn")
            return
        if self.dual_ctrl is None:
            self.log("도봇 미연결", "err")
            return
        def _run():
            self.robot_busy = True
            try:
                self.log("에러 초기화 — 대기점 복귀 중...", "info")
                self.dual_ctrl.recover_to_standby()
                self.root.after(0, lambda: self._set_badge(self._badge_robot_a, "연결됨", COLOR_OK))
                self.root.after(0, lambda: self._set_badge(self._badge_robot_b, "연결됨", COLOR_OK))
            except Exception as e:
                self.log(f"에러 초기화 오류: {e}", "err")
            finally:
                self.robot_busy = False
        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # 초기 보드 세팅
    # ─────────────────────────────────────────────────────────
    def _reset_board(self) -> None:
        self.board.reset()
        self._draw_board()
        self.log("보드 초기 배치 리셋 완료", "ok")

    # ─────────────────────────────────────────────────────────
    # 모델 불러오기
    # 로드 완료 후 _try_upgrade_to_vision_ctrl() 자동 호출
    # ─────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="체스 모델 선택 (YOLO .pt 또는 CNN .pt)",
            filetypes=[("PyTorch 모델", "*.pt *.pth"), ("모든 파일", "*.*")],
        )
        if not path:
            return
        def _run():
            self.log(f"모델 로드 중: {path}", "info")
            try:
                from chess_vision import _is_yolo_model
                name = path.split("/")[-1]
                if _is_yolo_model(path):
                    self.vision._load_yolo(path)
                    badge_text = "YOLO OK"
                else:
                    self.vision._load_cnn(path)
                    badge_text = "CNN OK"
                if self.vision.dummy_mode:
                    self.log("모델 로드 실패 — 더미 모드 유지", "err")
                    return
                config.MODEL_PATH = path
                self.model_path_var.set(name)
                self.root.after(0, lambda: self._set_badge(self._badge_vision, badge_text, COLOR_OK))
                self.log(f"모델 로드 완료: {name} ({badge_text})", "ok")
                self._try_upgrade_to_vision_ctrl()
            except Exception as e:
                self.log(f"모델 로드 실패: {e}", "err")
        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # 캘리브레이션
    # 완료 후 _try_upgrade_to_vision_ctrl() 자동 호출
    # ─────────────────────────────────────────────────────────
    def _recalib(self) -> None:
        if self.vision is None:
            self.log("비전 미초기화", "warn")
            return
        def _fetch():
            try:
                frame = self.vision._grab()
            except Exception as e:
                self.log(f"캘리브레이션 오류: {e}", "err")
                return
            self.root.after(0, lambda: self._open_calib_window(frame))
        threading.Thread(target=_fetch, daemon=True).start()

    def _open_calib_window(self, frame) -> None:
        LABELS   = ["h8 좌상", "h1 우상", "a1 우하", "a8 좌하"]
        points   = []
        scale_xy = [1.0, 1.0]

        win = tk.Toplevel(self.root)
        win.title("캘리브레이션 — h8→h1→a1→a8 순서로 클릭")
        win.configure(bg=BG)
        win.grab_set()

        guide_var = tk.StringVar(value=f"① 클릭:  {LABELS[0]}  (0 / 4)")
        tk.Label(win, textvariable=guide_var, bg=BG, fg=COLOR_INFO,
                 font=HFONT).pack(pady=(10, 4))

        canvas = tk.Canvas(win, bg="#111", cursor="crosshair", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        btn_row = tk.Frame(win, bg=BG)
        btn_row.pack(fill="x", padx=10, pady=(0, 10))
        btn_undo   = tk.Button(btn_row, text="↩ 되돌리기", bg="#333", fg=FG,
                               font=FONT, relief="flat")
        btn_undo.pack(side="left", padx=(0, 6), ipadx=6, ipady=4)
        btn_cancel = tk.Button(btn_row, text="✗ 취소", bg="#5c1a1a", fg=COLOR_ERR,
                               font=FONT, relief="flat")
        btn_cancel.pack(side="right", ipadx=6, ipady=4)

        orig_h, orig_w = frame.shape[:2]
        DISP_W, DISP_H = 900, 600
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img    = Image.fromarray(rgb).resize((DISP_W, DISP_H), Image.LANCZOS)
        imgtk  = ImageTk.PhotoImage(image=img)
        scale_xy[0] = orig_w / DISP_W
        scale_xy[1] = orig_h / DISP_H
        win.geometry(f"{DISP_W + 20}x{DISP_H + 100}")
        canvas.config(width=DISP_W, height=DISP_H)
        canvas.create_image(0, 0, anchor="nw", image=imgtk)
        canvas.imgtk = imgtk

        COLORS = ["#00ff88", "#ffdd00", "#ff6688", "#44bbff"]

        def _redraw():
            canvas.delete("marker")
            for i, (px, py) in enumerate(points):
                dx, dy = px / scale_xy[0], py / scale_xy[1]
                canvas.create_oval(dx-10, dy-10, dx+10, dy+10,
                                   outline=COLORS[i], width=2, tags="marker")
                canvas.create_text(dx+14, dy-12, text=LABELS[i],
                                   fill=COLORS[i], font=("Consolas", 10, "bold"),
                                   anchor="w", tags="marker")

        def _update_guide():
            n = len(points)
            if n < 4:
                guide_var.set(f"{'①②③④'[n]} 클릭:  {LABELS[n]}  ({n} / 4)")
            else:
                guide_var.set("4점 완료 — 저장 중...")

        def _on_click(event):
            if len(points) >= 4:
                return
            points.append((event.x * scale_xy[0], event.y * scale_xy[1]))
            _redraw()
            _update_guide()
            if len(points) == 4:
                win.after(400, _finish)

        canvas.bind("<Button-1>", _on_click)
        btn_undo.config(command=lambda: (points.pop(), _redraw(), _update_guide()) if points else None)
        btn_cancel.config(command=lambda: (self.log("캘리브레이션 취소", "warn"), win.destroy()))
        win.protocol("WM_DELETE_WINDOW", win.destroy)

        def _finish():
            try:
                src = np.array(points, dtype=np.float32)
                W   = 800
                dst = np.array([[0,0],[W,0],[W,W],[0,W]], dtype=np.float32)
                self.vision.H, _ = cv2.findHomography(src, dst)
                self.vision._save_calib()
                self.log("캘리브레이션 완료 및 저장", "ok")

                # ── vision_robot_calib.json 자동 생성 ──
                # 캘리브 순서: h8(좌상) → h1(우상) → a1(우하) → a8(좌하)
                pixel_pts = [
                    (0,   0  ),   # h8
                    (800, 0  ),   # h1
                    (800, 800),   # a1
                    (0,   800),   # a8
                ]
                mm_pts = [
                    square_to_mm_for(Robot.A, 'h8'),
                    square_to_mm_for(Robot.A, 'h1'),
                    square_to_mm_for(Robot.A, 'a1'),
                    square_to_mm_for(Robot.A, 'a8'),
                ]
                calib = VisionRobotCalib()
                calib.calibrate(pixel_pts, mm_pts, robot_label="A")
                # 컨트롤러에 즉시 반영
                if isinstance(self.dual_ctrl, VisionGuidedController):
                    self.dual_ctrl.calib = calib
                self.log("비전 좌표 캘리브 자동 생성 완료", "ok")
                self.root.after(0, self._try_upgrade_to_vision_ctrl)
            except Exception as e:
                self.log(f"캘리브레이션 저장 오류: {e}", "err")
            finally:
                win.destroy()

    def _capture_dataset(self) -> None:
        """현재 카메라 프레임을 dataset/ 폴더에 저장."""
        import os
        from datetime import datetime
        if self.vision is None:
            self.log("비전 미초기화", "warn")
            return
        try:
            os.makedirs("dataset", exist_ok=True)
            frame = self.vision._grab()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = f"dataset/{ts}.jpg"
            cv2.imwrite(path, frame)
            # 현재까지 저장된 총 개수
            count = len([f for f in os.listdir("dataset") if f.endswith(".jpg")])
            self.log(f"캡처 저장: {path}  (총 {count}장)", "ok")
        except Exception as e:
            self.log(f"캡처 오류: {e}", "err")

    # ─────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────
    def _reconnect_vision(self) -> None:
        def _run():
            try:
                self.log("비전 재연결 중...", "info")
                self.root.after(0, lambda: self._set_badge(
                    self._badge_vision, "재연결 중...", COLOR_WARN))
                if self.vision:
                    self.vision.release()
                self.vision = ChessVision()
                status = "더미모드" if self.vision.dummy_mode else "모델OK"
                color  = COLOR_WARN if self.vision.dummy_mode else COLOR_OK
                self.root.after(0, lambda: self._set_badge(self._badge_vision, status, color))
                self.log(f"비전 재연결 완료 ({status})", "ok")
                self._try_upgrade_to_vision_ctrl()
            except Exception as e:
                self.log(f"비전 재연결 오류: {e}", "err")
                self.root.after(0, lambda: self._set_badge(
                    self._badge_vision, "오류", COLOR_ERR))
        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # Stockfish 재연결
    # ─────────────────────────────────────────────────────────
    def _reconnect_stockfish(self) -> None:
        def _run():
            try:
                self.log("Stockfish 재연결 중...", "info")
                self.root.after(0, lambda: self._set_badge(self._badge_sf, "재연결 중...", COLOR_WARN))
                if self.engine:
                    self.engine.quit()
                self.engine = ChessEngine()
                self.root.after(0, lambda: self._set_badge(self._badge_sf, "준비됨", COLOR_OK))
                self.log("Stockfish 재연결 완료", "ok")
            except Exception as e:
                self.log(f"Stockfish 재연결 오류: {e}", "err")
                self.root.after(0, lambda: self._set_badge(self._badge_sf, "없음", COLOR_ERR))
        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # 로봇 재연결
    # ─────────────────────────────────────────────────────────
    def _reconnect(self, robot: Robot) -> None:
        if self.robot_busy:
            self.log("도봇 사용 중", "warn")
            return
        if self.dual_ctrl is None:
            self.log("도봇 미연결", "err")
            return

        def _run():
            self.robot_busy = True
            rs    = self.dual_ctrl._get_rs(robot)
            badge = self._badge_robot_a if robot == Robot.A else self._badge_robot_b
            try:
                self.log(f"[Robot {robot.value}] 재연결 중...", "info")
                self.root.after(0, lambda: self._set_badge(badge, "재연결 중...", COLOR_WARN))
                if rs.device:
                    try:
                        rs.device.close()
                    except Exception:
                        pass
                    rs.device = None
                self.dual_ctrl._init_robot(rs)
                self.root.after(0, lambda: self._set_badge(badge, "연결됨", COLOR_OK))
                self.log(f"[Robot {robot.value}] 재연결 완료", "ok")
            except Exception as e:
                self.log(f"[Robot {robot.value}] 재연결 오류: {e}", "err")
                self.root.after(0, lambda: self._set_badge(badge, "오류", COLOR_ERR))
            finally:
                self.robot_busy = False

        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # 종료
    # ─────────────────────────────────────────────────────────
    def on_close(self) -> None:
        self.cam_running = False
        if self.engine:
            self.engine.quit()
        if self.vision:
            self.vision.release()
        if self.dual_ctrl:
            self.dual_ctrl.quit()
        self.root.destroy()


# ─────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────
def main() -> None:
    root = tk.Tk()
    root.geometry(config.WINDOW_SIZE)
    app  = ChessRobotGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()