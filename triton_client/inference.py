import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
#import ultralytics.utils.triton as triton
from torchvision import transforms
import torch
import  numpy as np
import cv2


class InferenceClient(grpcclient.InferenceServerClient):
    def __init__(self, triton_server_ip, endpoint, scheme, triton=True, imgsz=640, labels=None, class_num = 7) -> None:
        #self.establish_connection(scheme, triton_server_ip)
        # if scheme=="http":
        #     self.scheme_class = httpclient
        #     httpclient.InferenceServerClient.__init__(self, url=triton_server_ip+":8000")
        # if scheme=="grpc":
        #     self.scheme_class = grpcclient
        #     grpcclient.InferenceServerClient.__init__(self, url=triton_server_ip+":8001")
        if triton:
            super().__init__(url=triton_server_ip+":8001")
        else:
            model=self.as_edge()
        self.scheme_class = grpcclient
        self.triton_ip = triton_server_ip
        self.endpoint = endpoint
        self.scheme  = scheme
        self.class_num = class_num

        self.inputs = None
        self.outputs = None
        
        self.labels = self.fill_labels(labels)

        # self.inputs = grpcclient.InferInput("input__0", [3, 224, 224], datatype="FP32")
        # self.outputs = grpcclient.InferRequestedOutput("output__0", class_count=7)
        #super().__init__(url=self.triton_ip)
        #self.triton_ip = triton_server_ip+":"+port
        

    def establish_connection(self, scheme, triton_ip):
        if scheme=="http":
            triton_ip = triton_ip+":8000"
            self.scheme_class = httpclient
            return httpclient.InferenceServerClient().__init__(self, url=triton_ip)
        elif scheme=="grpc":
            triton_ip = triton_ip+":8001"
            self.scheme_class = grpcclient
            return grpcclient.InferenceServerClient().__init__(self, url=triton_ip)

    def call(self, image: np.ndarray):
        #transformed_image = self.preprocess(image)
        transformed_image = self.preprocess(image)
        if 'cls' in self.endpoint:
            if self.inputs ==None and self.outputs ==None:
                self.inputs = self.scheme_class.InferInput("input__0", transformed_image.shape, datatype="FP32")
                self.outputs = self.scheme_class.InferRequestedOutput("output__0", class_count= self.class_num)
        # if classification:
        #     parameters = [{
        #         "classification": { int64_param : 2 }
        #     }]
        #     self.outputs{}
        #self.inputs.set_data_from_numpy(transformed_image)
            self.inputs.set_data_from_numpy(transformed_image)
            results = self.infer(model_name=self.endpoint, inputs=[self.inputs], outputs=[self.outputs])
            if self.class_num<5:
                inference_output = results.as_numpy("output__0")
            else:
                inference_output = results.as_numpy("output__0")[:5]
            
            return inference_output
        elif self.endpoint=="trash-classification":
            if self.inputs ==None and self.outputs ==None:
                self.inputs = self.scheme_class.InferInput("input__0", transformed_image.shape, datatype="FP32")
                self.outputs = self.scheme_class.InferRequestedOutput("output__0", class_count= self.class_num)

            self.inputs.set_data_from_numpy(transformed_image)
            results = self.infer(model_name=self.endpoint, inputs=[self.inputs], outputs=[self.outputs])
            if self.class_num<5:
                inference_output = results.as_numpy("output__0")
            else:
                inference_output = results.as_numpy("output__0")[:5]
            
            return inference_output
    def fill_labels(self, labels):
        try:
            with open(labels) as f: return eval(f.read())
        except Exception as e:
            print(e)
    def match_classes(self, byte_strings : list[bytes]):
        preds = {}
        for byte_string in byte_strings:
            output = byte_string.decode("utf-8")
            confidence, classification = output.split(":")
            preds[self.labels[int(classification)]] = float(confidence)
        return preds

    def match_class(self, byte_string: bytes):

        output = byte_string.decode("utf-8")
        confidence, classification = output.split(":")
        return (confidence,self.labels[int(classification)])
        

    # def inputs(self):
    #     pass
    def crop_center(self, img,cropx,cropy):
        y,x,c = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx, :]
    
    def preprocess(self, image: np.ndarray):
        width, length, c = image.shape
        if width !=640  or length != 640:
            cropped = cv2.resize(image, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
        else:
            cropped = image

        if 'cls' in self.endpoint or self.endpoint=="trash-classification":
            resized = cv2.resize(cropped, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        else:
            resized = cropped

        # if resized.shape != (3, 640, 640) or resized.shape != (3, 224,  224):
        #     print("shifted axies")
        #     axes_shifted = np.moveaxis(resized, 2, 0)
        # else:
        #     axes_shifted = resized
        transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32)
                        ]) 
  
        # Convert the image to Torch tensor 
        image_tensor = transform(resized) 
        
        
        #return axes_shifted.astype(np.float32)
        #print(image_tensor)
        t = torch.tensor(image_tensor)
        return t.numpy()

        # if not isinstance(image, np.ndarray):
        #     preprocess = transforms.Compose(
        #         [
        #         transforms.Resize(224),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #         ]
        #     )
        #     return preprocess(image).numpy()
        # else:
        #     return image
        
    def as_edge(self):
        pass


    # def _close(self):
    #     #return self.triton_client.close()
    #     self.close()
    #     return


if __name__ == "__main__":
    model = InferenceClient("192.168.191.129", "yolov8l-cls", "grpc", 8001)
    print(model.triton_client.is_server_live())
    print(model)

    model.__call__(np.asarray())



    model.triton_client.close()