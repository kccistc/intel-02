from iotdemo import FactoryController

# Example usage:
ctrl = FactoryController(port='/dev/ttyACM0')  # Initialize the FactoryController with port '/dev/ttyACM0'

while True:
    input_number = int(input("number from 1 to 8"))
    if input_number == 1:
        ctrl.system_start()
        print("system on")
    elif input_number == 2:
        ctrl.system_stop()
        print("system off")
    elif input_number == 3:
        ctrl.red ^= False
    elif input_number == 4:
        ctrl.orange ^= False
    elif input_number == 5:
        ctrl.green ^= False
    elif input_number == 6:
        ctrl.conveyor ^= False
    elif input_number == 7:
        ctrl.push_actuator(1)
    elif input_number == 8:
        ctrl.push_actuator(2)
    elif input_number == 9:
        print("End")
        break

# Close the FactoryController when done
ctrl.close()
