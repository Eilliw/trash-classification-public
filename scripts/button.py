import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
from gpiozero import Servo,pins
import dotenv
import os
import sys
import time
from picamera2 import Picamera2
from functools import partial
from PIL import Image
import cv2



os.chdir(".")
cwd = os.getcwd()
print(cwd)
sys.path.insert(0, f"{cwd}/triton_client")
sys.path.insert(0, f"{cwd}/edge_testing/tools")
from inference import InferenceClient
from dataset import PiCamDataset





pigpio_factory = pins.pigpio.PiGPIOFactory()
#must start pigpio daemon with sudo pigpiod
dotenv.load_dotenv(".env")

def next_path(path_pattern):
    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b

class InferenceButton():
    def __init__(self, inference_client_object :InferenceClient , picam2_obj, servo1, servo2, mode="run") -> None:
        self.mode = mode
        self.inference_client_obj = inference_client_object
        self.picam2_obj = picam2_obj
        
        self.picamdataset = PiCamDataset()
        
        GPIO.setwarnings(False) # Ignore warning for now
        GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
        GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)

        self.servo_pulse_widths = [.5/1000,0, 2.5/1000]
        #GPIO.add_event_detect(10,GPIO.RISING,callback=self.button_callback,bouncetime=100)
        self.servo1 = Servo(servo1, min_pulse_width=self.servo_pulse_widths[0], max_pulse_width= self.servo_pulse_widths[2], pin_factory=pigpio_factory)
        self.servo2 = Servo(servo2, min_pulse_width=self.servo_pulse_widths[0], max_pulse_width= self.servo_pulse_widths[2], pin_factory=pigpio_factory)
        
        #setting servos to be initially in their mid position to lock trash can lid
        print("setting servos to their initial mid position")
        time.sleep(.25)
        for servo in [self.servo1, self.servo2]:
            servo.mid()
            servo.detach()
            time.sleep(.2)

        #message = input("Press enter to quit\n\n")
    def capture_img(self, picam2):
        frame = picam2.capture_array("main")
        print(frame.shape)
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #img comes out in 4:3 ratio - we are resizing it to a square to not distort the image
        cropped = client.crop_center(rgb_img, 1232,1232)
        #scaling down the picture to usable dims
        scaled_down = cv2.resize(cropped, dsize=(640,640), interpolation=cv2.INTER_CUBIC)
        image = scaled_down
            
        return image
    def inference_request(self, image, all_outputs=False):
        client = self.inference_client_obj
        output = client.call(image)
        if not all_outputs:
            clean_output = client.match_class(output[0])
            return clean_output
        else:
            clean_outputs: dict = client.match_classes(output)
            return clean_outputs
    def check_if_different_enough(self, t, r, difference=15):
        #normalize
        t = t*100
        r = r*100
        dif = abs(t-r)
        if dif > difference:
            return True
        else:
            return False
    def button_callback(self, channel):
        ### send inference request to InfrenceClient
        image = self.capture_img(self.picam2_obj)
        #output = self.inference_request(image)
        outputs: dict =self.inference_request(image, all_outputs=True)
        ### take inference  output and move matching servo
        #output = (1.0000, 'Trash')
        print("Button was pushed!")
        print(f"detections: {outputs}")
        highest_classification_key = max(outputs, key=outputs.get)
        print(f"What was detected: {highest_classification_key}-{outputs[highest_classification_key]}")
        
        #check if classification is different enough
        try:
            if self.check_if_different_enough(outputs["Trash"], outputs["Recycle"]):
                pass
            else:
                print("model classification score is not different enough, allowing user override")
                #moving both servos
                self.servo1.max()
                time.sleep(.2)
                self.servo1.detach()
                self.servo2.max()
                time.sleep(.2)
                self.servo2.detach()
                print("waiting 10 seconds")
                time.sleep(10)
                self.servo1.mid()
                time.sleep(.2)
                self.servo1.detach()
                self.servo2.mid()
                time.sleep(.2)
                self.servo2.detach()
                img = Image(cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC))
                if os.listdir("lib/unlabeled-imgs") < 5000:
                    PiCamDataset.save_to_local(img, path="lib/unlabeled-imgs", path_pattern=None, sequential=False)
                return
        except:
            print("check if different couldn't complete")

        classification = highest_classification_key
        #print(f"detected : {output[1]}-{output[0]}")
        #classification = output[1]
        if classification =='Trash' or classification=='Recycle':
            pass
        elif classification in ['Paper', 'Metal', 'Plastic', 'Cardboard', 'Glass']:
            classification = 'Recycle'
        elif classification in ['Trash', 'Compost']:
            classification ='Recycle'
        if classification == 'Trash':
            servo = self.servo1
        elif classification == "Recycle":
            servo = self.servo2
        #print(f"servo value : {servo.value}\nservo pusle width:{servo.pulse_width}")
        #print(f"classification as [{classification}-{output[0]}]")
        
        #servo.min()
        #sleep(1)
        #print(f"servo min val:{servo.value}")
        #servo.mid()
        #sleep(1)
        #print(f"servo mid val:{servo.value}")
        #servo.max()
        #sleep(1)
        #print(f"servo max val:{servo.value}")
        servo.max()
        print(f"servo min val:{servo.value}\nsleeping for 10 seconds")
        time.sleep(10)
        servo.mid()
        print(f"servo min val:{servo.value}")
        time.sleep(.5)
        servo.detach()
        
        if self.mode == 'test':
            self.test(image, classification)
    def test(self, image, classification):
        print("entering test phase")
        class_q = input(f"was the class prediction of {classification} correct?(y/n): ")
        im = Image.fromarray(image)
        if class_q == 'yes':
            next_name = next_path(f"{cwd}/lib/test_collection/{classification}/{classification}-%s.jpg")
            im.save(next_name)
            print(f"saved test image to {next_name}")
        else:
            match classification:
                case "Trash":
                    classification="Recycle"
                case 'Recycle':
                    classification="Trash"
            next_name = next_path(f"{cwd}/lib/test_collection/{classification}/{classification}-%s.jpg")
            im.save(next_name)
            print(f"saved test image to {next_name}")
        return
    def clean_up(self):
        GPIO.cleanup()
    

#GPIO.setwarnings(False) # Ignore warning for now
#GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
#GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)

#GPIO.add_event_detect(10,GPIO.RISING,callback=button_callback) # Setup event on pin 10 rising edge

#servo_pulse_widths = [1/1000,0, 2/1000]

#servo = Servo(27, min_pulse_width=servo_pulse_widths[0], max_pulse_width=servo_pulse_widths[2])
#message = input("Press enter to quit\n\n") # Run until someone presses enter
if __name__ == "__main__":
    
    print(os.getcwd())
    client = InferenceClient("192.168.191.129", "trash-classification", "grpc", labels='lib/labels/2classes.txt')
    print("made client")
    print("aliveness check:", client.is_server_live())
    
    picam2 = Picamera2()
    
    still_config = picam2.create_still_configuration(main={"size": (640,640), "format":'RGB888'})
    smallest_full_res_config = picam2.create_still_configuration(main={"size": (1640, 1232), "format":'RGB888'})
    picam2.configure(smallest_full_res_config)
    picam2.start()
    time.sleep(1)
    
    button = InferenceButton(client, picam2, 27, 22, mode='run')
    
    #initilising camera
    
    try:
        while True:
            #capturing image
            
            #manual button check
            input_state = GPIO.input(10)
            if input_state == True:
                button.button_callback(None)
            time.sleep(.1)
            pass
            
    except KeyboardInterrupt:
        button.picamdataset.collection_upload(path="lib/unlabeled-imgs", delete=True)
        button.clean_up()
        picam2.stop()
        client.close()
    
    

#GPIO.cleanup() # Clean up
