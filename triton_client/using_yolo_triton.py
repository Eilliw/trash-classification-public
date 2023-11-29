import ultralytics.utils.triton as triton
import time

#pip install tritonclient\[all\]

IP = "192.168.191.129"
#model = YOLO(IP+":8000/trash_detection", task="detect")
model = triton.TritonRemoteModel(IP+":8001",endpoint="yolov8l-cls", scheme='grpc')
print(model.triton_client.is_server_live())

print(model.input_names)

time.sleep(5)
model.triton_client.close()
