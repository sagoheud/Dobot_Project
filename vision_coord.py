"""
vision_coord.py
---------------
비전 픽셀 좌표 ↔ 도봇 mm 좌표 실시간 변환 모듈.

핵심 아이디어:
  카메라가 본 체스판 픽셀 좌표를 도봇 mm 좌표로 직접 변환.
  판이 조금 틀어지거나 말이 비껴있어도 비전이 실제 위치를 잡아줌.

캘리브레이션 방법:
  1) 체스판 위 4개 코너에 도봇 엔드이펙터를 직접 갖다대서 mm 좌표 측정
  2) 같은 위치를 카메라로 찍어서 픽셀 좌표 측정
  3) 4쌍 대응점으로 변환 행렬 계산 → JSON 저장

이후엔 카메라 캡처 한 번으로 임의 칸의 도봇 mm 좌표가 실시간으로 나옴.
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config

FILES = list("abcdefgh")
RANKS = list("12345678")

CALIB_SAVE_PATH = "vision_robot_calib.json"


# ─────────────────────────────────────────────────────────────
# 캘리브레이션 — 픽셀 ↔ 도봇 mm 변환 행렬
# ─────────────────────────────────────────────────────────────
class VisionRobotCalib:
    """
    카메라 픽셀 좌표 ↔ 도봇 mm 좌표 상호 변환.

    캘리브레이션 포인트:
        체스판 4 코너를 도봇으로 실측한 mm + 카메라로 찍은 픽셀 쌍 4개.
        → cv2.getPerspectiveTransform으로 3×3 변환 행렬 계산.

    저장 형식 (vision_robot_calib.json):
    {
        "robot": "A",               # 어떤 로봇 기준인지
        "pixel_pts": [[px,py], ...],  # 카메라 픽셀 4점 (800×800 워핑 기준)
        "mm_pts":    [[x,y], ...],    # 도봇 mm 4점
        "M_px2mm": [[...], ...],      # 픽셀→mm 3×3 행렬
        "M_mm2px": [[...], ...]       # mm→픽셀 역행렬
    }
    """

    def __init__(self, path: str = CALIB_SAVE_PATH):
        self.path = Path(path)
        self.M_px2mm: Optional[np.ndarray] = None  # 픽셀 → mm
        self.M_mm2px: Optional[np.ndarray] = None  # mm → 픽셀 (역행렬)
        self.robot_label: str = "A"
        self._load()

    @property
    def is_ready(self) -> bool:
        return self.M_px2mm is not None

    # ── 캘리브레이션 실행 ───────────────────────────────────
    def calibrate(
        self,
        pixel_pts: list[tuple],
        mm_pts: list[tuple],
        robot_label: str = "A",
    ) -> None:
        """
        픽셀-mm 대응 4쌍으로 변환 행렬 계산 후 저장.

        Parameters
        ----------
        pixel_pts : [(px,py) × 4]  — 워핑 이미지(800×800) 기준 픽셀 좌표
                    순서: a8(좌상) → h8(우상) → h1(우하) → a1(좌하)
        mm_pts    : [(x,y) × 4]    — 대응하는 도봇 mm 좌표 (로봇A or B 기준)
        robot_label : "A" or "B"
        """
        assert len(pixel_pts) == 4 and len(mm_pts) == 4, "4점 필요"

        src = np.array(pixel_pts, dtype=np.float32)
        dst = np.array(mm_pts, dtype=np.float32)

        # getPerspectiveTransform: 정확히 4쌍 → 정확한 행렬
        self.M_px2mm = cv2.getPerspectiveTransform(src, dst)
        self.M_mm2px = cv2.getPerspectiveTransform(dst, src)
        self.robot_label = robot_label

        self._save(pixel_pts, mm_pts)
        print(f"[VisionCalib] 캘리브레이션 완료 (Robot {robot_label})")

    # ── 변환 함수 ───────────────────────────────────────────
    def pixel_to_mm(self, px: float, py: float) -> tuple[float, float]:
        """워핑 픽셀 좌표 → 도봇 mm 좌표."""
        if self.M_px2mm is None:
            raise RuntimeError("캘리브레이션 먼저 실행 필요")
        pt = np.array([[[px, py]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self.M_px2mm)
        x, y = res[0][0]
        return (round(float(x), 1), round(float(y), 1))

    def mm_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """도봇 mm 좌표 → 워핑 픽셀 좌표 (디버그용)."""
        if self.M_mm2px is None:
            raise RuntimeError("캘리브레이션 먼저 실행 필요")
        pt = np.array([[[x, y]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self.M_mm2px)
        px, py = res[0][0]
        return (round(float(px), 1), round(float(py), 1))

    # ── 저장/로드 ───────────────────────────────────────────
    def _save(self, pixel_pts, mm_pts) -> None:
        data = {
            "robot": self.robot_label,
            "pixel_pts": [list(p) for p in pixel_pts],
            "mm_pts": [list(p) for p in mm_pts],
            "M_px2mm": self.M_px2mm.tolist(),
            "M_mm2px": self.M_mm2px.tolist(),
        }
        self.path.write_text(json.dumps(data, indent=2))
        print(f"[VisionCalib] 저장: {self.path}")

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        self.M_px2mm = np.array(data["M_px2mm"], dtype=np.float64)
        self.M_mm2px = np.array(data["M_mm2px"], dtype=np.float64)
        self.robot_label = data.get("robot", "A")
        print(f"[VisionCalib] 로드 완료: {self.path} (Robot {self.robot_label})")


# ─────────────────────────────────────────────────────────────
# 비전 기반 실시간 좌표 계산
# ─────────────────────────────────────────────────────────────
class VisionCoordResolver:
    """
    카메라 한 번 찍어서 특정 칸의 도봇 mm 좌표를 실시간으로 계산.

    흐름:
        square name → 워핑 이미지 내 칸 중심 픽셀 → VisionRobotCalib → mm 좌표

    특징:
        - 호모그래피(H)가 이미 체스판을 정사각형으로 펴줬으므로
          칸 중심 픽셀 계산이 단순 분할로 해결됨
        - 말이 칸 안에서 살짝 치우쳐 있어도 칸 중심으로 이동 → 안전함
        - 비전 신뢰도(CNN softmax 최대값) 반환 → 낮으면 fallback
    """

    WARP_SIZE = 800  # chess_vision.py의 _warp() 와 맞춰야 함

    def __init__(
        self,
        vision,          # ChessVision 인스턴스
        calib: VisionRobotCalib,
        confidence_thresh: float = 0.70,  # 이 값 미만이면 fallback 권고
    ):
        self.vision = vision
        self.calib = calib
        self.confidence_thresh = confidence_thresh

    # ── 칸 중심 픽셀 계산 ───────────────────────────────────
    def square_to_pixel(self, square: str) -> tuple[float, float]:
        """
        체스 표기 → 워핑 이미지(800×800) 내 칸 중심 픽셀.
        호모그래피로 이미 정사각형화됐으므로 단순 분할로 계산 가능.
        """
        file = square[0].lower()
        rank = square[1]
        col = FILES.index(file)         # a=0, h=7
        row = 7 - RANKS.index(rank)     # rank1=row7(하단), rank8=row0(상단)

        cell = self.WARP_SIZE / 8
        px = col * cell + cell / 2      # 칸 중심 X 픽셀
        py = row * cell + cell / 2      # 칸 중심 Y 픽셀
        return (px, py)

    # ── 핵심 API: 칸 이름 → 도봇 mm ─────────────────────────
    def resolve(self, square: str) -> tuple[float, float]:
        """
        square → 카메라 촬영 → 도봇 mm 좌표.

        Returns
        -------
        (x_mm, y_mm)  도봇 이동 좌표

        Raises
        ------
        RuntimeError : 캘리브레이션 미완료
        """
        if not self.calib.is_ready:
            raise RuntimeError(
                "VisionRobotCalib 캘리브레이션이 완료되지 않았습니다.\n"
                "run_calibration_wizard()를 먼저 실행하세요."
            )

        px, py = self.square_to_pixel(square)
        x_mm, y_mm = self.calib.pixel_to_mm(px, py)
        return (x_mm, y_mm)

    # ── CNN confidence 확인 포함 버전 ───────────────────────
    def resolve_with_confidence(
        self,
        square: str,
        expected_label: Optional[str] = None,
    ) -> dict:
        """
        칸 이름 → mm 좌표 + CNN 신뢰도 검사.

        Parameters
        ----------
        square         : "e2" 등
        expected_label : "wP", "bK" 등 — 예상 기물 레이블 (None이면 검사 생략)

        Returns
        -------
        {
            "mm":         (x, y),         # 도봇 이동 좌표
            "pixel":      (px, py),        # 워핑 이미지 픽셀 좌표
            "label":      "wP",            # CNN 인식 결과
            "confidence": 0.94,            # softmax 최댓값
            "ok":         True,            # 신뢰도 & 레이블 일치 여부
            "warn":       ""               # 경고 메시지 (ok=False 시)
        }
        """
        if not self.calib.is_ready:
            raise RuntimeError("VisionRobotCalib 캘리브레이션 필요")

        px, py = self.square_to_pixel(square)
        x_mm, y_mm = self.calib.pixel_to_mm(px, py)

        # CNN 추론 (단일 칸)
        label, confidence = self._infer_single_cell(square)

        ok = True
        warn = ""

        if confidence < self.confidence_thresh:
            ok = False
            warn = f"신뢰도 낮음 ({confidence:.0%}) — fallback 권고"

        if expected_label and label != expected_label:
            ok = False
            warn += f" | 레이블 불일치 (기대:{expected_label}, 인식:{label})"

        return {
            "mm":         (x_mm, y_mm),
            "pixel":      (px, py),
            "label":      label,
            "confidence": confidence,
            "ok":         ok,
            "warn":       warn.strip(),
        }

    # ── 내부: 단일 칸 CNN 추론 ──────────────────────────────
    def _infer_single_cell(self, square: str) -> tuple[str, float]:
        """워핑 이미지에서 단일 칸 CNN 추론 → (label, confidence)."""
        import torch

        if self.vision.model is None or self.vision.dummy_mode:
            return ("unknown", 0.0)

        frame = self.vision._grab()
        warped = self.vision._warp(frame)

        col = FILES.index(square[0].lower())
        row = 7 - RANKS.index(square[1])
        cell_img = self.vision._crop_cell(warped, row, col)
        cell_rgb = cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB)

        from chess_vision import _INFER_TRANSFORM
        tensor = _INFER_TRANSFORM(cell_rgb).unsqueeze(0).to(self.vision.device)

        with torch.no_grad():
            logits = self.vision.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            idx = probs.argmax().item()
            conf = probs[idx].item()

        label = config.PIECE_CLASSES[idx]
        return (label, conf)


# ─────────────────────────────────────────────────────────────
# 캘리브레이션 위저드 (최초 1회 실행)
# ─────────────────────────────────────────────────────────────
def run_calibration_wizard(
    vision,
    robot_label: str = "A",
    calib_path: str = CALIB_SAVE_PATH,
) -> VisionRobotCalib:
    """
    터미널 안내 + 카메라 미리보기로 캘리브레이션 4점 수집.

    실행 방법:
        vision = ChessVision(...)          # chess_vision.py
        calib = run_calibration_wizard(vision, robot_label="A")

    수집 포인트 (로봇A 기준):
        a8 좌상 → h8 우상 → h1 우하 → a1 좌하
        각 코너에서:
          ① 도봇 조그로 엔드이펙터를 칸 중심에 갖다댐
          ② get_pose()로 (x, y) mm 읽어서 입력
          ③ 워핑 이미지 창에서 같은 코너 픽셀 클릭

    로봇B 기준이라면 h1 → a1 → a8 → h8 순으로 수집 후
    robot_label="B" 로 저장. 로봇B 좌표계 반전은 square_to_mm_for()
    에서 처리하므로 여기선 로봇B 기준 실측값 그대로 입력.
    """
    corners = ["a8 (좌상)", "h8 (우상)", "h1 (우하)", "a1 (좌하)"]
    pixel_pts = []
    mm_pts = []

    print("=" * 60)
    print(f"  비전↔도봇 캘리브레이션 위저드  [Robot {robot_label}]")
    print("=" * 60)

    # ① 워핑 이미지 생성
    frame = vision._grab()
    warped = vision._warp(frame)

    # ② 각 코너 클릭 수집
    clicked = []
    clone = warped.copy()
    corner_labels_cv = ["a8", "h8", "h1", "a1"]

    def _on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append((x, y))
            cv2.circle(clone, (x, y), 6, (0, 255, 100), -1)
            cv2.putText(
                clone,
                corner_labels_cv[len(clicked) - 1],
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 100),
                2,
            )
            print(f"  픽셀 클릭: {corners[len(clicked)-1]} → ({x}, {y})")

    win = "캘리브레이션 — 4 코너 순서대로 클릭"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _on_click)

    print("\n[1단계] 워핑 이미지 창에서 코너를 순서대로 클릭하세요:")
    for i, c in enumerate(corners):
        print(f"  {i+1}. {c}")

    while len(clicked) < 4:
        disp = clone.copy()
        n = len(clicked)
        if n < 4:
            msg = f"클릭 {n+1}/4: {corners[n]}"
            cv2.putText(
                disp, msg, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2
            )
        cv2.imshow(win, disp)
        if cv2.waitKey(30) == 27:
            cv2.destroyWindow(win)
            raise RuntimeError("캘리브레이션 취소")

    cv2.destroyWindow(win)
    pixel_pts = clicked

    # ③ 각 코너의 도봇 mm 좌표 수동 입력
    print("\n[2단계] 각 코너에서 도봇 get_pose()로 읽은 XY mm 입력:")
    print("  (도봇 조그로 엔드이펙터를 칸 중심에 정확히 위치시킨 뒤 읽으세요)\n")

    for i, c in enumerate(corners):
        while True:
            try:
                raw = input(f"  {c}  X Y (공백 구분, 예: 192.0 -104.5): ").strip()
                x, y = map(float, raw.split())
                mm_pts.append((x, y))
                print(f"    → ({x}, {y}) mm 등록")
                break
            except ValueError:
                print("    형식 오류 — 'X Y' 로 입력하세요")

    # ④ 행렬 계산 및 저장
    calib = VisionRobotCalib(path=calib_path)
    calib.calibrate(pixel_pts, mm_pts, robot_label=robot_label)

    # ⑤ 검증 출력
    print("\n[검증] 4 코너 역변환 오차:")
    for i, (pp, mp) in enumerate(zip(pixel_pts, mm_pts)):
        recalc_mm = calib.pixel_to_mm(*pp)
        err = (
            abs(recalc_mm[0] - mp[0]),
            abs(recalc_mm[1] - mp[1]),
        )
        print(f"  {corners[i]}: 입력({mp[0]:.1f},{mp[1]:.1f})  "
              f"역산({recalc_mm[0]:.1f},{recalc_mm[1]:.1f})  "
              f"오차({err[0]:.2f},{err[1]:.2f})mm")

    print("\n캘리브레이션 완료. 이제 vision_guided_pick()을 사용할 수 있습니다.")
    return calib


# ─────────────────────────────────────────────────────────────
# 통합 픽앤플레이스 좌표 결정 함수
# ─────────────────────────────────────────────────────────────
def vision_guided_mm(
    square: str,
    resolver: VisionCoordResolver,
    fallback_fn,          # square_to_mm_for(robot, square) 형태 callable
    fallback_robot,       # Robot.A or Robot.B
    expected_label: Optional[str] = None,
    log_fn=print,
) -> tuple[float, float]:
    """
    비전 기반 mm 좌표 계산. 신뢰도 낮으면 fallback 좌표 사용.

    Parameters
    ----------
    square         : "e2" 등 체스 표기
    resolver       : VisionCoordResolver 인스턴스
    fallback_fn    : square_to_mm_for 함수 (기존 하드코딩 좌표)
    fallback_robot : fallback 시 사용할 로봇
    expected_label : 예상 기물 레이블 (None이면 CNN 확인 생략, 속도 우선)

    Returns
    -------
    (x_mm, y_mm)
    """
    try:
        if expected_label is not None:
            result = resolver.resolve_with_confidence(square, expected_label)
            if result["ok"]:
                log_fn(
                    f"[VisionCoord] {square.upper()} → "
                    f"({result['mm'][0]:.1f}, {result['mm'][1]:.1f}) mm  "
                    f"conf={result['confidence']:.0%}  [{result['label']}]"
                )
                return result["mm"]
            else:
                log_fn(
                    f"[VisionCoord] {square.upper()} 비전 실패 → fallback  "
                    f"({result['warn']})"
                )
        else:
            # confidence 확인 없이 좌표만 빠르게 계산 (PLACE 위치 등)
            mm = resolver.resolve(square)
            log_fn(
                f"[VisionCoord] {square.upper()} → ({mm[0]:.1f}, {mm[1]:.1f}) mm"
            )
            return mm

    except Exception as e:
        log_fn(f"[VisionCoord] 오류 → fallback  ({e})")

    # Fallback: 기존 하드코딩 좌표
    fb = fallback_fn(fallback_robot, square)
    log_fn(f"[VisionCoord] {square.upper()} fallback → ({fb[0]:.1f}, {fb[1]:.1f}) mm")
    return fb
