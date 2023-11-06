from iotdemo import FactoryController

Led_ctrl = FactoryController('/dev/ttyACM0')
flag = True

def main():

    while (True):
        number = int(input("LED NUM: "))
        if number == 1:
            Led_ctrl.system_start()
        elif number == 2:
            Led_ctrl.system_stop()
        elif number == 3:
           flag = Led_ctrl.red
           Led_ctrl.red = flag
        elif number == 4:
            flag = Led_ctrl.orange
            Led_ctrl.orange = flag
        elif number == 5:
            flag = Led_ctrl.green
            Led_ctrl.green = flag
        elif number == 6:
            flag = Led_ctrl.conveyor
            Led_ctrl.conveyor = flag
        elif number == 7:
            Led_ctrl.push_actuator(1)
        elif number == 8:
            Led_ctrl.push_actuator(2)
        else :
            break;

    Led_ctrl.close()
if __name__ == "__main__":
    main()
