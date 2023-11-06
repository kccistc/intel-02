# hw1 Arduino와 Python을 이용한 LED 제어

from iotdemo import FactoryController
ctrl = FactoryController('/dev/ttyACM0')

while True:
    data = int(input("input number: "))

    if data == 1:
        ctrl.system_start()
    
    elif data == 2:
        ctrl.system_stop()
        break

    elif data == 3:
        flag = ctrl.red
        ctrl.red = flag

    elif data == 4:
        flag = ctrl.orange
        ctrl.orange = flag

    elif data == 5:
        flag = ctrl.green
        ctrl.green = flag

    elif data == 6:
        flag = ctrl.conveyor
        ctrl.conveyor = flag

    elif data == 7:
        ctrl.push_actuator(1)

    elif data == 8:
        ctrl.push_actuator(2)

ctrl.close()

