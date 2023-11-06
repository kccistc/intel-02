import sys
import os
from iotdemo import FactoryController
# FactoryController를 초기화하고 시스템을 시작합니다.
ctrl = FactoryController('/dev/ttyACM0')

# 다음으로 사용할 API 번호를 입력받기
choice = int(input("Select an API (1-8): "))

# 선택한 API 실행
if choice == 1:
    ctrl.red = True
    ctrl.orange = False
    ctrl.green = False
elif choice == 2:
    ctrl.red = False
    ctrl.orange = False
    ctrl.green = False
elif choice == 3:
    ctrl.red = True
    ctrl.orange = False
    ctrl.green = False
elif choice == 4:
    ctrl.red = False
    ctrl.orange = True
    ctrl.green = False
elif choice == 5:
    ctrl.red = False
    ctrl.orange = False
    ctrl.green = True
elif choice == 6:
    ctrl.conveyor = not ctrl.conveyor
elif choice == 7:
    ctrl.push_actuator(1)
elif choice == 8:
    ctrl.push_actuator(2)
else:
    print("Invalid choice")

# FactoryController를 닫습니다.
ctrl.close()

