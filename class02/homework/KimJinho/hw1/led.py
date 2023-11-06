from iotdemo import FactoryController

ctrl = FactoryController('/dev/ttyACM0')


while True:
    ex = True
    n = int(input("input number: "))
    if n==1:
        ctrl.system_start()
    if n==2:
        ex = False
    elif n==3:
        print("Red Toggle")
        flag = ctrl.red
        ctrl.red = flag
    elif n==4:
        print("Orange Toggle")
        flag = ctrl.orange
        ctrl.orange = flag
    elif n==5:
        print("Green Toggle")
        flag = ctrl.green
        ctrl.green = flag
    elif n==6:
        print("conveyor")
        flag = ctrl.conveyor
        ctrl.conveyor = flag
    elif n==7:
        print("actuator 1")
        ctrl.push_actuator(1)
    elif n==8:
        print("actuator 2")
        ctrl.push_actuator(2)
    if ex == False:
        ctrl.system_stop()
        break
ctrl.close()
