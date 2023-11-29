from inference import InferenceClient
import numpy as np
import cv2
import torch
from tritonclient.utils import triton_to_np_dtype
from torchvision import transforms
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import time

CLASSIFICATION_LABELS = ["Cardboard",
"Compost",
"Glass",
"Metal",
"Paper",
"Plastic",
"Trash"]
CLASSIFICATION_LABELS= []
with open('lib/labels/yolov8-7classes.txt') as f: CLASSIFICATION_LABELS = eval(f.read())
with open('lib/labels/imagenet1000.txt') as f: image_net_labels = eval(f.read())
INPUT_SIZE = (224,224,3)

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx, :]

if __name__ == "__main__":
    print(CLASSIFICATION_LABELS)
    print(image_net_labels)
    client = InferenceClient("192.168.191.129", "yolov8l-cls", "grpc", labels='lib/labels/yolov8-7classes.txt')
    #client = httpclient.InferenceServerClient("192.168.191.129:8001")
    print("made client")
    print(client.is_server_live())
    # print(model.triton_client.is_server_live())
    # print(model)
    
    # print(model.input_formats)
    # print(model.output_names)
    # print(model.np_input_formats)
    # print(model.triton_client.get_model_config("yolov8l-cls"))

    print('hello')
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    time.sleep(2)
    if cap.isOpened():
        print("cap is opened")
        while True:
            ret, frame = cap.read()
            if ret:
                print(frame.shape)
                # top_left_corner = (frame.shape[0]/2 -(INPUT_SIZE[0]/2), frame.shape[1]/2 -(INPUT_SIZE[1]/2)
                # bottom_right_corner = (frame.shape[0]/2 +(INPUT_SIZE[0]/2), frame.shape[1]/2 +(INPUT_SIZE[1]/2)
                #resized_frame = np.resize(frame, (64,3,3,3))
                #snp.float32
                #frame_tensor = np.resize(frame, (224,224,3))
                #frame_tensor =  torch.rand(1, 3, 640, 640, dtype=torch.float32)
                #frame_tensor  = np.resize(frame, (16,3,320,640))
                #frame_tensor = np.resize(frame, (224, 1280, 3))
                #np_frame = np.array(frame)
                #frame_tensor = np.reshape(frame, (3, 224, 224))
                cropped = crop_center(frame, 640,640)
                # resized = cv2.resize(cropped, (224,224))
                # resized_axes_shifted = np.moveaxis(resized, 2, 0)
                # img = resized_axes_shifted.astype(np.float32)
                
                model_infer = client.call(frame)
                
                print(model_infer)
                # inference = model_infer[0].decode("utf-8")
                # print(inference, type(inference))
                # first_inference = inference.split(':')
                # print(first_inference)
                #classification = image_net_labels[int(first_inference[1])]
                output = client.match_class(model_infer[0])
                print(output)
                
                cv2.imshow('Color Frame', cropped)
                
                key = cv2.waitKey(1)
                
                if key == ord('q'):
                    break
            else:
                print('Frame not available')
                print(cap.isOpened())
        
    else:
        print("video not opened")
    client.close()
    cap.release()
    cv2.destroyAllWindows()