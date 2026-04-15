"""
chess_vision.py
---------------
탑뷰 카메라 → 체스판 인식 → 보드 상태 반환.

담당 기능:
  - 카메라 캡처
  - 호모그래피 캘리브레이션
  - YOLO 기반 말 탐지 (실제 위치 추적) — best.pt / chess_cnn.pt 이름 무관
  - ResNet18 CNN 기반 분류 (YOLO 없을 때 fallback)
  - 다중 프레임 투표로 인식 안정화
"""

import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

import config

FILES = list("abcdefgh")
RANKS = list("12345678")

# YOLO 클래스명 → 내부 레이블 변환
YOLO_TO_LABEL = {
    "black-bishop": "bB", "black-king":   "bK", "black-knight": "bN",
    "black-pawn":   "bP", "black-queen":  "bQ", "black-rook":   "bR",
    "white-bishop": "wB", "white-king":   "wK", "white-knight": "wN",
    "white-pawn":   "wP", "white-queen":  "wQ", "white-rook":   "wR",
}


# ─────────────────────────────────────────────────────────────
# ResNet18 CNN (YOLO 없을 때 fallback)
# ─────────────────────────────────────────────────────────────
class ChessCNN(nn.Module):
    def __init__(self, num_classes: int = len(config.PIECE_CLASSES)):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


_INFER_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((config.CNN_INPUT_SIZE, config.CNN_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _is_yolo_model(path: str) -> bool:
    """파일 헤더에 ultralytics 문자열 있으면 YOLO 모델."""
    try:
        with open(path, "rb") as f:
            return b"ultralytics" in f.read(8192)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# 메인 클래스
# ─────────────────────────────────────────────────────────────
class ChessVision:
    """
    Parameters
    ----------
    model_path  : YOLO 또는 ResNet18 .pt 파일 경로
    camera_index: cv2.VideoCapture 인덱스
    calib_path  : 캘리브레이션 저장/로드 경로 (JSON)
    device      : 'cpu' | 'cuda'
    """

    def __init__(
        self,
        model_path: str = config.MODEL_PATH,
        camera_index: int = config.CAMERA_INDEX,
        calib_path: str = config.CALIB_PATH,
        board_mm: float = config.BOARD_MM,
        origin_mm: tuple = (config.ORIGIN_X_MM, config.ORIGIN_Y_MM),
        model: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        self.camera_index = camera_index
        self.calib_path   = Path(calib_path)
        self.board_mm     = board_mm
        self.origin_mm    = np.array(origin_mm, dtype=float)
        self.cell_mm      = board_mm / 8.0
        self.device       = torch.device(device)
        self.dummy_mode   = False
        self.yolo_mode    = False          # YOLO 모드 여부
        self.H: Optional[np.ndarray] = None

        self.model      = None
        self.yolo_model = None

        self._load_model(model, model_path)
        self._load_calib()
        self._open_camera()

    # ── 모델 로드 ────────────────────────────────────────────
    def _load_model(self, model: Optional[nn.Module], model_path: str) -> None:
        if model is not None:
            self.model = model.to(self.device).eval()
            return

        path = Path(model_path)
        if not path.exists():
            self.dummy_mode = True
            print("[Vision] ⚠ 더미 모드: 모델 없음")
            return

        if _is_yolo_model(model_path):
            self._load_yolo(model_path)
        else:
            self._load_cnn(model_path)

    def _load_yolo(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(model_path)
            self.yolo_mode  = True
            self.dummy_mode = False
            print(f"[Vision] YOLO 모델 로드 완료: {model_path}")
        except Exception as e:
            print(f"[Vision] YOLO 로드 실패: {e} → 더미 모드")
            self.dummy_mode = True

    def _load_cnn(self, model_path: str) -> None:
        try:
            self.model = ChessCNN(len(config.PIECE_CLASSES)).to(self.device)
            state = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                if not any(k.startswith("backbone.") for k in state.keys()):
                    state = {"backbone." + k: v for k, v in state.items()}
                self.model.load_state_dict(state)
            else:
                self.model = state.to(self.device)
            self.model.eval()
            self.dummy_mode = False
            print(f"[Vision] CNN 모델 로드 완료: {model_path}")
        except Exception as e:
            print(f"[Vision] CNN 로드 실패: {e} → 더미 모드")
            self.dummy_mode = True

    # ── 카메라 ──────────────────────────────────────────────
    def _open_camera(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"카메라 {self.camera_index} 열기 실패 — "
                "ls /dev/video* 로 인덱스 확인 후 config.py 수정"
            )
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_CONTRAST,   config.CAMERA_CONTRAST)
        self.cap.set(cv2.CAP_PROP_SATURATION, config.CAMERA_SATURATION)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, config.CAMERA_BRIGHTNESS)
        for _ in range(5):
            self.cap.read()
        print(f"[Vision] 카메라 {self.camera_index} 연결됨")

    def _grab(self) -> np.ndarray:
        frame = None
        for _ in range(10):
            ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("카메라 프레임 읽기 실패")
        return frame

    # ── 캘리브레이션 ────────────────────────────────────────
    def _save_calib(self) -> None:
        self.calib_path.write_text(json.dumps({"H": self.H.tolist()}))

    def _load_calib(self) -> None:
        if self.calib_path.exists():
            data = json.loads(self.calib_path.read_text())
            self.H = np.array(data["H"])
            print(f"[Vision] 캘리브레이션 로드: {self.calib_path}")

    # ── 이미지 변환 ─────────────────────────────────────────
    def _warp(self, frame: np.ndarray, size: int = 800) -> np.ndarray:
        if self.H is None:
            raise RuntimeError("캘리브레이션 먼저 실행")
        return cv2.warpPerspective(frame, self.H, (size, size))

    def _crop_cell(self, warped: np.ndarray, row: int, col: int, margin: float = 0.1) -> np.ndarray:
        H, W = warped.shape[:2]
        cell_h, cell_w = H // 8, W // 8
        y0 = row * cell_h + int(cell_h * margin)
        y1 = (row + 1) * cell_h - int(cell_h * margin)
        x0 = col * cell_w + int(cell_w * margin)
        x1 = (col + 1) * cell_w - int(cell_w * margin)
        return warped[y0:y1, x0:x1]

    # ── YOLO 추론 ────────────────────────────────────────────
    def _infer_yolo(self, frame: np.ndarray) -> dict:
        """
        YOLO로 말 탐지 → {square: (label, cx_pixel, cy_pixel)} 반환.
        cx, cy는 원본 프레임 기준 픽셀 좌표.
        """
        results = self.yolo_model(frame, verbose=False)[0]
        detections = {}

        if self.H is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.yolo_model.names[cls_id]
            conf = float(box.conf[0])
            if conf < 0.4:
                continue
            label = YOLO_TO_LABEL.get(cls_name)
            if label is None:
                continue

            # 바운딩박스 중심점 (원본 픽셀)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = y2 - (y2 - y1) * 0.1

            # 호모그래피로 워핑 픽셀 좌표로 변환
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            warped_pt = cv2.perspectiveTransform(pt, self.H)[0][0]
            wx, wy = warped_pt

            cell = 800 / 8
            col = int(wx // cell)        # wx: 수평(a~h), 0=a열, 7=h열
            row = int(wy // cell)        # wy: 수직(8~1), 0=rank8, 7=rank1
            if 0 <= col <= 7 and 0 <= row <= 7:
                file = FILES[col]
                rank = RANKS[7 - row]   # row=0 → rank8, row=7 → rank1
                square = f"{file}{rank}"
                # 같은 칸에 여러 탐지 시 confidence 높은 것 선택
                conf = float(box.conf[0])
                if square not in detections or conf > detections[square][2]:
                    detections[square] = (label, wx, wy, conf)
        if detections:
            print(f"[YOLO 탐지] {detections.keys()}")
            for sq, (lbl, wx, wy, conf) in detections.items():
                print(f"  {sq}: {lbl} 픽셀({wx:.0f},{wy:.0f}) conf={conf:.2f}")
        return detections

    def get_piece_pixel(self, square: str) -> Optional[tuple]:
        """
        YOLO로 특정 칸의 말 실제 픽셀 위치 반환 (워핑 이미지 기준).
        말이 칸 안에서 치우쳐 있어도 실제 위치 반환.

        Returns
        -------
        (wx, wy) : 워핑 이미지 픽셀 좌표  또는  None (탐지 실패)
        """
        if not self.yolo_mode or self.H is None:
            return None

        frame = self._grab()
        detections = self._infer_yolo(frame)
        if square in detections:
            _, wx, wy, _ = detections[square]
            return (wx, wy)
        return None

    # ── CNN 추론 (fallback) ──────────────────────────────────
    def _infer_batch(self, warped: np.ndarray) -> list:
        if self.dummy_mode or self.model is None:
            return ["empty"] * 64

        tensors = []
        for row in range(8):
            for col in range(8):
                cell = self._crop_cell(warped, row, col)
                cell_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
                tensors.append(_INFER_TRANSFORM(cell_rgb))

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            indices = self.model(batch).argmax(dim=1).tolist()
        return [config.PIECE_CLASSES[i] for i in indices]

    # ── 보드 상태 반환 ───────────────────────────────────────
    def get_board(self, n_frames: int = config.VISION_N_FRAMES) -> dict:
        """
        보드 전체 상태 반환 {"a1": "wP", ...}.
        YOLO 모드면 YOLO로, 아니면 CNN으로 추론.
        """
        if self.dummy_mode:
            print("[Vision] 더미 모드 — 빈 보드 반환")
            return {}

        if self.yolo_mode:
            return self._get_board_yolo(n_frames)
        else:
            return self._get_board_cnn(n_frames)

    def _get_board_yolo(self, n_frames: int) -> dict:
        votes: dict[str, dict[str, int]] = {}
        for _ in range(n_frames):
            frame = self._grab()
            detections = self._infer_yolo(frame)
            for sq, (label, *_) in detections.items():
                if sq not in votes:
                    votes[sq] = {}
                votes[sq][label] = votes[sq].get(label, 0) + 1
            time.sleep(0.05)

        board = {}
        for sq, vote in votes.items():
            board[sq] = max(vote, key=vote.get)
        return board

    def _get_board_cnn(self, n_frames: int) -> dict:
        votes: list[dict[str, int]] = [{} for _ in range(64)]
        for _ in range(n_frames):
            frame = self._grab()
            warped = self._warp(frame)
            labels = self._infer_batch(warped)
            for i, lbl in enumerate(labels):
                votes[i][lbl] = votes[i].get(lbl, 0) + 1
            time.sleep(0.05)

        board = {}
        for i, vote in enumerate(votes):
            label = max(vote, key=vote.get)
            if label == "empty":
                continue
            row, col = i // 8, i % 8
            rank = RANKS[7 - row]
            file = FILES[col]
            board[f"{file}{rank}"] = label
        return board

    def release(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()