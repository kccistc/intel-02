from iotdemo import FactoryController
#from factory_controller import FactoryController
ctrl = FactoryController('/dev/tty/ACMO')
while (True):
    a= int(input())
    if a==1:
        ctrl.system_start()
        print("System start...")
    elif a==2:
        ctrl.system_stop()
        print("System stopped...")
        break
    elif a==3:
        #flag = ctrl.red
        #ctrl.red = flag
        ctrl.red = True
    elif a==4:
        #flag = ctrl.orange
        #ctrl.orange = flag
        ctrl.orange = True
    elif a==5:
        #flag = ctrl.green
        #ctrl.green = flag
        ctrl.green = True
    elif a==6:
        #flag = ctrl.conveyor
        #ctrl.conveyor = flag
        ctrl.conveyor = True
    elif a==7:
        ctrl.push_actuator(1)
    elif a==8:
        ctrl.push_actuator(2)
ctrl.close()
