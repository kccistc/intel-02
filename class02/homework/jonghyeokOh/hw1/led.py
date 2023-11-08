from factory_controller import FactoryController
#import FactoryController

def func(ctrl):
    Chk_systemStart = False;
    print("hi1")
    while True :
        input_ = input("typing~:")
        if (input_ == 'q'):
            break
        else:
            input_ = int(input_)

        if (Chk_systemStart == False):
            if (input_ == 1):
                ctrl.system_start()
                Chk_systemStart = True
                print("activate ctrl")
            else:
                print("Wrong cmd. start factory controller first")
        else:
            if (input_ == 1):
                print("Wrong cmd. controller is already running")
            elif (input_ == 2):
                ctrl.system_stop()
                Chk_systemStart = False
                print("shutdown ctrl")
                break
            elif (input_ == 3):
                print("try toggle")
                ctrl.red ^= False
            elif (input_ == 4):
                ctrl.orange ^= False
            elif (input_ == 5):
                ctrl.green ^= True
            elif (input_ == 6):
                ctrl.conveyor = not ctrl.conveyor
            elif (input_ == 7):
                ctrl.push_actuator(1)
            else: #input_ == 8
                ctrl.push_actuator(2)


def main():
    ctrl = FactoryController('/dev/ttyACM1')
    #FactoryController fc
    func(ctrl)
    ctrl.close()


if __name__ == "__main__":
    print("f")
    main()

