from iotdemo import FactoryController

ctrl =FactoryController('/dev/ttyACM0')
status = True
while(1):
    a=int(input())
    if(a==1):
        ctrl.system_start()
    elif(a==2):
        ctrl.system_stop()
        ctrl.close()
        break
    elif(a==3):
        status = ctrl.red
        ctrl.red = status
    elif(a==4):
        status = ctrl.orange
        ctrl.orange = status
    elif(a==5):
        status = ctrl.green
        ctrl.green = status
    elif(a==6):
        status = ctrl.conveyor
        ctrl.conveyor = status
    elif(a==7):
        ctrl.push_actuator(1)
    elif(a==8):
        ctrl.push_actuator(2)
ctrl.close()
