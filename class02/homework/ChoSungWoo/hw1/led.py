#from iotdemo import FactoryController
#__all__ = ('FactoryController')
#ctrl = FactoryController('/dev/ttyACM0')

from pyfirmata import Arduino
import time

arduino = Arduino('dev/ttyACM0')

btn1 = arduino.get_pin()
btn2 = arduino.get_pin()
btn3 = arduino.get_pin()
btn4 = arduino.get_pin()

while True:
	
	if (btn1==0) :
		
	elif (btn1==1) :
		
	if (btn1==0) :
	
	elif (btn1==1) :
		
	if (btn1==0) :
	
	elif (btn1==1) :
		
	if (btn1==0) :

	elif (btn1==1) :
		
