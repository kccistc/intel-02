from iotdemo import FactoryController

def main():
    while(True):
        n = int(input("input number:"))
        if n == 1:
            ctrl.system_start()
        elif n == 2:
            ctrl.system_stop()
        elif n == 3:
            if ctrl.red is False:
                ctrl.red = False
            else:
                ctrl.red = True
        elif n == 4:
            if ctrl.orange is False:
                ctrl.orange = False
            else:
                ctrl.orange = True
        elif n == 5:
            if ctrl.green is False:
                ctrl.green = False
            else:
                ctrl.green = True
        elif n == 6:
            if ctrl.conveyor is False:
                ctrl.conveyor = False
            else:
                ctrl.conveyor = True
        elif n == 7:
            ctrl.push_actuator(1)
        elif n == 8:
            ctrl.push_actuator(2)
        else:
            break

ctrl = FactoryController('/dev/ttyACM0')

if __name__ == "__main__":
    main()
    ctrl.close()
