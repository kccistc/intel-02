from iotdemo import FactoryController

ctrl = FactoryController('/dev/ttyACM0')

flag = True

def main():
    while(True):
        API = int(input("API number(1~8) : "))
        if API == 1:
            ctrl.system_start()
        elif API == 2:
            ctrl.system_stop()
        elif API == 3:
            flag = ctrl.red
            ctrl.red = flag
        elif API == 4:
            flag = ctrl.orange
            ctrl.orange = flag
        elif API == 5:
            flag = ctrl.green
            ctrl.green = flag
        elif API == 6:
            flag = ctrl.conveyor
            ctrl.conveyor = flag
        elif API == 7:
            ctrl.push_actuator(1)
        elif API == 8:
            ctrl.push_actuator(2)
        elif API == 9:
            break

    ctrl.close()

if __name__ == "__main__":
    main()
