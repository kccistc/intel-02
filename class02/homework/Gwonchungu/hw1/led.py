from iotdemo import FactoryController

ctrl = FactoryController(port='/dev/ttyACM0')  # 실제 포트로 변경

while True:
    try:
        user_input = int(input("Enter a number between 1 and 8: "))
        if user_input == 1:
            ctrl.system_start()
        elif user_input == 2:
            ctrl.system_stop()
            break
        elif user_input == 3:
            ctrl.red ^= False
        elif user_input == 4:
            ctrl.orange ^= False
        elif user_input == 5:
            ctrl.green ^= False
        elif user_input == 6:
            ctrl.conveyor ^= False
        elif user_input == 7:
            ctrl.push_actuator(1)
        elif user_input == 8:
            ctrl.push_actuator(2)
        else:
            print("Invalid input! Enter a number between 1 and 8.")
    except ValueError:
        print("Invalid input! Enter a number between 1 and 8.")


ctrl.close()
