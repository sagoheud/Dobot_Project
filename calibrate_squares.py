"""
calibrate_squares.py
--------------------
로봇A : rank8 (a8~h8) 좌표 기록
로봇B : rank1 (a1~h1) 좌표 기록

사용법:
  1) 로봇을 조그로 목표 칸 중심 위에 정확히 위치
  2) 엔터 → 현재 XY 기록
  3) 완료 후 결과를 config.py 에 붙여넣기
"""

import time
import json
from pydobot import Dobot
import config

# ─────────────────────────────────────────────────────────────
FILES = list("abcdefgh")

def connect(port: str) -> Dobot:
    print(f"  연결 중: {port} ...", end=" ", flush=True)
    dev = Dobot(port=port, verbose=False)
    print("OK")
    return dev

def get_xy(dev: Dobot) -> tuple:
    p = dev.pose()
    return round(p[0], 2), round(p[1], 2)

def record_rank(dev: Dobot, robot_label: str, squares: list) -> dict:
    result = {}
    print()
    for sq in squares:
        input(f"  [{robot_label}] {sq.upper()} 위에 조그로 맞추고 엔터 ▶ ")
        x, y = get_xy(dev)
        result[sq] = (x, y)
        print(f"    → {sq.upper()} = ({x}, {y})")
    return result

# ─────────────────────────────────────────────────────────────
print("=" * 50)
print("  체스 로봇 좌표 캘리브레이션")
print("=" * 50)

# 로봇A — rank8 (a8 ~ h8)
print("\n[로봇A] rank8 캘리브레이션 (a8 → h8)")
dev_a = connect(config.DOBOT_PORT_A)
squares_a = [f"{f}8" for f in FILES]   # a8, b8, c8, d8, e8, f8, g8, h8
coords_a = record_rank(dev_a, "Robot A", squares_a)
dev_a.close()

# 로봇B — rank1 (a1 ~ h1)
print("\n[로봇B] rank1 캘리브레이션 (a1 → h1)")
dev_b = connect(config.DOBOT_PORT_B)
squares_b = [f"{f}1" for f in FILES]   # a1, b1, c1, d1, e1, f1, g1, h1
coords_b = record_rank(dev_b, "Robot B", squares_b)
dev_b.close()

# ─────────────────────────────────────────────────────────────
# 결과 출력
print("\n" + "=" * 50)
print("  기록 완료 — config.py 에 아래 값 붙여넣기")
print("=" * 50)

print("\n# 로봇A rank8 실측 좌표")
print("COORD_A_RANK8 = {")
for sq, (x, y) in coords_a.items():
    print(f'    "{sq}": ({x}, {y}),')
print("}")

print("\n# 로봇B rank1 실측 좌표")
print("COORD_B_RANK1 = {")
for sq, (x, y) in coords_b.items():
    print(f'    "{sq}": ({x}, {y}),')
print("}")

# JSON 저장
out = {"robot_a_rank8": coords_a, "robot_b_rank1": coords_b}
with open("calib_result.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\n→ calib_result.json 에도 저장됨")

# """
# calib_b_a열.py
# --------------
# 로봇B 기준 a1 ~ a5 좌표 기록.
# 조그로 맞추고 엔터 누르면 기록됨.
# """

# import time
# from pydobot import Dobot
# import config

# SQUARES = ["a1", "a2", "a3", "a4", "a5"]

# print("=" * 40)
# print("  로봇B a1~a5 좌표 기록")
# print("=" * 40)
# print(f"  연결 중: {config.DOBOT_PORT_B} ...", end=" ", flush=True)
# dev = Dobot(port=config.DOBOT_PORT_B, verbose=False)
# print("OK")

# results = {}
# for sq in SQUARES:
#     input(f"\n  [{sq.upper()}] 위에 조그로 맞추고 엔터 ▶ ")
#     p = dev.pose()
#     x, y = round(p[0], 2), round(p[1], 2)
#     results[sq] = (x, y)
#     print(f"    → {sq.upper()} = ({x}, {y})")

# dev.close()

# print("\n" + "=" * 40)
# print("  결과")
# print("=" * 40)
# for sq, (x, y) in results.items():
#     print(f'    "{sq}": ({x}, {y}),')

# # 현재 계산값과 비교
# print("\n=== 현재 계산값 vs 실측값 ===")
# BOARD_MM_B = config.BOARD_MM_B
# _CELL_B = BOARD_MM_B / 8
# x1_cur, y1_cur = results["a1"]
# print(f"{'칸':4} {'계산X':8} {'실측X':8} {'차X':6}")
# for sq, (rx, ry) in results.items():
#     r = int(sq[1])
#     cx = round(x1_cur + (r-1) * _CELL_B, 2)
#     print(f"{sq:4} {cx:8.2f} {rx:8.2f} {cx-rx:+6.2f}")