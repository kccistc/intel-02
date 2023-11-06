"""
Smart Factory HW module controller
"""

from iotdemo.factory_controller.factory_controller import FactoryController

ctrl = FactoryController('/dev/ttyACM0')

while True:
    key = int(input("key:"))
    
    if key == 1:
        ctrl.system_start()
        print("system_start")
        
    elif key == 2:
        ctrl.system_stop()
        print("system_stop")
        
    elif key == 3:
        status = ctrl.red
        ctrl.red = status
            
        print("green {}".format(ctrl.red))
        
    elif key == 4:
        status = ctrl.orange
        ctrl.orange = status
            
        print("green {}".format(ctrl.orange))
        
    elif key == 5:
        status = ctrl.green
        ctrl.green = status
            
        print("green {}".format(ctrl.green))
        
    elif key == 6:
        status = ctrl.conveyor
        ctrl.conveyor = status
        print("conveyor")
        
    elif key == 7:
        ctrl.push_actuator(1)
        print("push_actuator1")
        
    elif key == 8:
        ctrl.push_actuator(2)
        print("push_actuator2")
        
    else:
        break
        
ctrl.close()