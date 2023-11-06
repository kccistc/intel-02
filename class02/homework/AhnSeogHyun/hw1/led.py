import time
from iotdemo import FactoryController

ctrl = FactoryController('/dev/ttyACM0')

start_flag = False

if ctrl != 0:
    print("Arduino Connected.")
    while 1:
        user_input = input("Order : ")
        
        if user_input == '1':
            ctrl.system_start()
        
        elif user_input == '2':
            ctrl.system_stop()
            ctrl.close()
            break
        
        elif user_input == '3':
            ctrl.red ^= False
            
        elif user_input == '4':
            ctrl.orange ^= False
            
        elif user_input == '5':
            ctrl.green ^= False
            
        elif user_input == '6':
            ctrl.conveyor ^= False
            
        elif user_input == '7':
            ctrl.push_actuator(1)
        
        elif user_input == '8':
            ctrl.push_actuator(2)
    
else:
    print("Arduino is not conneted") 
