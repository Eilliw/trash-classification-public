from picamera2 import Picamera2
import time
from inference import InferenceClient
from button import InferenceButton
import numpy as np


def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx, :]

if __name__ == "__main__":
	#initilising triton client
    client = InferenceClient("192.168.191.129", "yolov8l-cls", "grpc", labels='lib/labels/yolov8-7classes.txt')
    print("made client")
    print("aliveness check:", client.is_server_live())
    
    button = InferenceButton(client, 27, 28, mode='run')
    
    #initilising camera
    picam2 = Picamera2()
	picam2.start()
	time.sleep(1)
	try:
		while True:
			#capturing image
			frame = picam2.capture_array("main")
			print(frame.shape)
			
			cropped = crop_center(frame, 640,640)
			
	except KeyboardInterrupt:
		button.clean_up()
		client.close()
		
	
	
	
		
	
