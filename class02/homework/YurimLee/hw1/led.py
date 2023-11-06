#!/usr/bin/env python

"""Module providing a Smart Factory HW module controller"""
from iotdemo.factory_controller import FactoryController


def main():
    """LED control using Arduino and Python"""

    # Init the system
    ctrl = FactoryController('/dev/ttyACM0')
    print("Please enter a number : 1 ~ 8")

    try:
        while True:
            input_cmd = int(input("Input : "))

            if input_cmd == 1:
                ctrl.system_start()
            elif input_cmd == 2:
                ctrl.system_stop()
            elif input_cmd == 3:
                ctrl.red = ctrl.red
            elif input_cmd == 4:
                ctrl.orange = ctrl.orange
            elif input_cmd == 5:
                ctrl.green = ctrl.green
            elif input_cmd == 6:
                ctrl.conveyor = ctrl.conveyor
            elif input_cmd == 7:
                ctrl.push_actuator(1)
            elif input_cmd == 8:
                ctrl.push_actuator(2)
            else:
                print("Please enter a valid number : 1 ~ 8")

    except KeyboardInterrupt:
        print("\nBye")

    finally:
        # Close the system
        ctrl.close()


if __name__ == "__main__":
    main()
