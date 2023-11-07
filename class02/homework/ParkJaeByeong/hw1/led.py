from iotdemo.factory_controller import FactoryController

ctrl = FactoryController('/dev/ttyACM0')

while True:
    user_input = input("Enter a command = ")
    
    if user_input == '1':
        ctrl.system_start()
        print("System started.")
    
    elif user_input == '2':
        ctrl.system_stop()
        print("System stopped.")
    
    elif user_input == '3':
        ctrl.red = False
        print("Red LED turned on.")
    
    elif user_input == '4':
        ctrl.orange = False
        print("Orange LED turned on")
    
    elif user_input == '5':
        ctrl.green = False
        print("Green LED turned on")
    
    elif user_input == '6':
        ctrl.conveyor = True
    
    elif user_input == '7':
        ctrl.push_actuator(1)
    
    elif user_input == '8':
        ctrl.push_actuator(2)
    
    elif user_input == 'q':
        break

ctrl.close()