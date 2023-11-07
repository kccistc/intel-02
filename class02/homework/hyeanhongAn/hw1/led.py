from iotdemo import FactoryController

# Initialize the system
with FactoryController('/dev/ttyACM0') as ctrl:
    input_number = 0 # Start the system

    try:
        while True:
            flag = True
            input_number = input("input num: ")

            if input_number == "1":
                ctrl.system_start()
                print("system_start....\r\n")
            elif input_number == "2":
                ctrl.system_stop()
                print("system_stop....\r\n")
            elif input_number == "3":
                flag = ctrl.red
                ctrl.red = flag
                #ctrl.red()
                print("red on\r\n")
            elif input_number == "4":
                flag = ctrl.orange
                ctrl.orange = flag
                #ctrl.orange()
                print("orange on\r\n")
            elif input_number == "5":
                flag = ctrl.green
                ctrl.green = flag
                #ctrl.green()
                print("green on\r\n")
            elif input_number == "6":
                flag = ctrl.conveyor
                ctrl.conveyor = flag
                print("conveyor on\r\n")
            elif input_number == "7":
                ctrl.push_actuator(1) 
                print("push_actuator\r\n")
            elif input_number == "8":
                ctrl.push_actuator(2)  
                print("push_actuator\r\n")
    except KeyboardInterrupt:
        pass 
    
    finally:
        ctrl.close()
