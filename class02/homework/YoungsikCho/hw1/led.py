"""
Smart Factory HW module controller
"""

import factory_controller

ctrl = factory_controller.FactoryController('/dev/ttyACM0')

while True:
    print("Enter a command (1-8, 9 to quit): ")
    command = int(input())
    if command == 1:
        ctrl.system_start()
    elif command == 2:
        ctrl.system_stop()
    elif command == 3:
        if ctrl.red is False:
            ctrl.red = False
        else:
            ctrl.red = True
        print(f" after red {ctrl.red}")
    elif command == 4:
        if ctrl.orange is False:
            ctrl.orange = False
        else:
            ctrl.orange = True
        print(f" after orange {ctrl.orange}")
    elif command == 5:
        if ctrl.green is False:
            ctrl.green = False
        else:
            ctrl.green = True
        print(f" after green {ctrl.green}")
    elif command == 6:
        if ctrl.conveyor is False:
            ctrl.conveyor = False
        else:
            ctrl.conveyor = True
        print(f" after conveyor {ctrl.conveyor}")
        
    elif command == 7:
        ctrl.push_actuator(1)
    elif command == 8:
        ctrl.push_actuator(2)
    elif command == 9:
        print("End")
        break

ctrl.close()
