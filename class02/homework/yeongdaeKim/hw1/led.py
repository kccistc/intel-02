from iotdemo import FactoryController

def solutionProc():
    ctrl = FactoryController("/dev/ttyACM0")
    if ctrl.is_dummy: return
    while True:
        n = int(input("Input Digit (Quit : -1) : "))
        # start status
        if n == 1:
            ctrl.system_start()
        # stop status
        elif n == 2:
            ctrl.system_stop()
        # red led on/off toggle
        elif n == 3:
            ctrl.red ^= False
        # orange status on/off toggle
        elif n == 4:
            ctrl.orange ^= False
        # green status on/off toggle
        elif n == 5:
            ctrl.green ^= False
        # conveyor status on/off toggle
        elif n == 6:
            ctrl.conveyor ^= False
        # button act 1
        elif n == 7:
            ctrl.push_actuator(1)
        # button act 2
        elif n == 8:
            ctrl.push_actuator(2)
        # quit
        elif n == -1:
            break
    ctrl.close()

solutionProc()

