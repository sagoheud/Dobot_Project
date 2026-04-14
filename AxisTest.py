"""
test_parking.py
---------------
각 로봇의 파킹 위치 좌표 읽기 테스트.

실행:
    python3 test_parking.py
"""

from pydobot import Dobot
import config

def read_pose(port: str, label: str):
    print(f"\n[{label}] {port} 연결 중...")
    try:
        dev = Dobot(port=port, verbose=False)
        p = dev.pose()
        x, y, z, r = p[0], p[1], p[2], p[3]
        print(f"[{label}] X={x:.1f}  Y={y:.1f}  Z={z:.1f}  R={r:.1f}")
        print(f"\n  config.py 에 넣을 값:")
        if label == "Robot A":
            print(f"  HOME_A_X = {x:.1f}")
            print(f"  HOME_A_Y = {y:.1f}")
            print(f"  HOME_A_R = {r:.1f}")
        else:
            print(f"  HOME_B_X = {x:.1f}")
            print(f"  HOME_B_Y = {y:.1f}")
            print(f"  HOME_B_R = {r:.1f}")
        dev.close()
    except Exception as e:
        print(f"[{label}] 오류: {e}")

print("=== 파킹 위치 측정 ===")
print("각 로봇을 원하는 파킹 자세로 조그로 맞춘 뒤 엔터를 누르세요.")

input("\n로봇A를 파킹 자세로 맞추고 엔터...")
read_pose(config.DOBOT_PORT_A, "Robot A")

input("\n로봇B를 파킹 자세로 맞추고 엔터...")
read_pose(config.DOBOT_PORT_B, "Robot B")

print("\n완료. 위 값을 config.py 에 입력하세요.")