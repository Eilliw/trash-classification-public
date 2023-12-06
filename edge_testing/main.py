from tools.dataset import PiCamDataset
from tools.gui import TestingGUI
from triton_client.inference  import InferenceClient
import dotenv, sys, cv2, threading,  time
from tools import dataset, gui

dotenv.load_dotenv(".env")


def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx, :]


def linux_image_stream():
    pass
def mac_image_stream(client  ,window: TestingGUI ,cond: threading.Condition):
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    time.sleep(2)
    if cap.isOpened():
        print("cap is opened")
        while True:
            cond.wait()

            #get image from camera
            ret, frame = cap.read()
            if ret:
                print(frame.shape)

                cropped = crop_center(frame, 640,640)
                #resized = cv2.resize(cropped, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
                #model  infer
                model_infer = client.call(cropped)

                #match classification
                output = client.match_class(model_infer[0])
                print(output)

                #send  classification  to window
                window.set_classification(output)
                #send  img to  window - this causes the window to aquire the lock
                window.set_canvas_img(cropped)

    else:
        print("video not opened")
if sys.platform == "linux":
    from picamera2 import Picamera2
elif  sys.platform == 'darwin':
    client = InferenceClient("192.168.191.129", "yolov8l-cls", "grpc", labels='lib/labels/yolov8-7classes.txt')
    #client = httpclient.InferenceServerClient("192.168.191.129:8001")
    print("made client")
    print(client.is_server_live())

    #get picam dataset
    dataset = PiCamDataset()
    
    cond = threading.Condition()

    window = TestingGUI(dataset, cond)
    image_stream = threading.Thread(target=mac_image_stream, args=(client, window, cond))

    window.root.protocol("WM_DELETE_WINDOW", window._on_close)
    
    image_stream.start()

    window.main_loop()
    image_stream.join()
    client.close()