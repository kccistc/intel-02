from iotdemo.factory_controller import FactoryController

ctrl = FactoryController('/dev/ttyACM6')

while True:
    num = input()
    if num == '1':
        ctrl.system_start() # input 1일때
    elif num == '2':
        ctrl.system_stop() # input 2일떄
    elif num == '3':
        red_flag = ctrl.red
        if red_flag:
            ctrl.red = True
        else:
            ctrl.red = False
    elif num == '4':
        orange_flag = ctrl.orange
        if orange_flag:
            ctrl.orange = True
        else:
            ctrl.orange = False
    elif num == '5':
        green_flag = ctrl.green
        if green_flag:
            ctrl.green = True
        else:
            ctrl.green = False
    elif num == '6':
        conveyor_flag = ctrl.conveyor
        if conveyor_flag:
            ctrl.conveyor = True
        else:
            ctrl.conveyor = False
    elif num == '7':
        ctrl.push_actuator(6)
    elif num == '8':
        ctrl.push_actuator(7)

ctrl.close()
