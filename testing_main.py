from edge_testing.tools.dataset import PiCamDataset
from edge_testing.tools.gui import TestingGUI
from triton_client.inference  import InferenceClient
from PIL import Image
import dotenv, sys, cv2, threading,  time


dotenv.load_dotenv(".env")

def get_client():
    return InferenceClient("192.168.191.129", "yolov8x-cls", "grpc", labels='lib/labels/yolov8-7classes.txt')

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx, :]


def linux_image_stream(window: TestingGUI, flag: threading.Event):
    #initilise client
    client = get_client()
    #client = httpclient.InferenceServerClient("192.168.191.129:8001")
    print("made client")
    print(client.is_server_live())
    picam2 = Picamera2()
    
    still_config = picam2.create_still_configuration(main={"size": (640,640), "format":'RGB888'})
    picam2.configure(still_config)
    picam2.start()
    time.sleep(1)
    while not window.exit_flag.is_set():
        flag.wait()
        try:
            frame = picam2.capture_array("main")
            print(frame.shape)
                
            cropped = client.crop_center(frame, 640,640)
            model_infer = client.call(cropped)

            #match classification
            output = client.match_class(model_infer[0])
            print(output)

            #send  classification  to window
            window.set_classification(output)
            #send  img to  window - this causes the window to aquire the lock
            window.set_canvas_img(cropped)
        except RuntimeError:
            print("Runtime error reached, most likely on purpose")
    picam2.stop()
    client.close()
def mac_image_stream(window: TestingGUI ,flag: threading.Event):
    #initilise client
    client = get_client()
    #client = httpclient.InferenceServerClient("192.168.191.129:8001")
    print("made client")
    print(client.is_server_live())

    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    time.sleep(2)
    if cap.isOpened():
        print("cap is opened")
        ret, frame = cap.read()
        # cropped = crop_center(frame, 640,640)
        # window.set_canvas_img(cropped)
        while not window.exit_flag.is_set():
            flag.wait()
            try:

                #get image from camera
                ret, frame = cap.read()
                if ret:
                    print(frame.shape)
                    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    

                    cropped = crop_center(rgb_img, 640,640)
                    #cropped = cv2.resize(rgb_img, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
                    ##model  infer
                    model_infer = client.call(cropped)
                    print(model_infer)

                    #match classification
                    output = client.match_class(model_infer[0])
                    print(output)

                    #send  classification  to window
                    window.set_classification(output)
                    #send  img to  window - this causes the window to clear the event flag
                    window.set_canvas_img(cropped)
            except RuntimeError:
                print("Runtime error reached, most likely on purpose")
        cap.release()
        cv2.destroyAllWindows()
        client.close()
    else:
        print("video not opened")

if __name__ == "__main__":
    

    #get picam dataset
    dataset = PiCamDataset()
    
    cond = threading.Event()

    window = TestingGUI(dataset, cond)
    if sys.platform == "linux":
        from picamera2 import Picamera2

    elif sys.platform == 'darwin':
        image_stream = threading.Thread(target=mac_image_stream, args=(window, cond))

    window.root.protocol("WM_DELETE_WINDOW", window._on_close)
    
    image_stream.start()

    window.root.mainloop()
    image_stream.join()