import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort

#import torch

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml

import sys
import importlib
import os
#importlib.__import__("/usr/lib/python3/dist-packages/libcamera", globals() )
#importlib.import_module('picamera2', '/usr/lib/python3/dist-packages/libcamera')

def apt_importer(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod
# spec = importlib.util.spec_from_file_location('libcamera', '/usr/lib/python3/dist-packages/libcamera/__init__.py')

# mod = importlib.util.module_from_spec(spec)
# #mod = importlib.abc.MetaPathFinder().find_module('libcamera',["/usr/lib/python3/dist-packages"])
# sys.modules[spec.name] = mod
# spec.loader.exec_module(mod)

pykms = apt_importer('pykms',"/usr/lib/python3/dist-packages/pykms/__init__.py")
libcamera = apt_importer('libcamera', '/usr/lib/python3/dist-packages/libcamera/__init__.py')
#DISTPACKAGES = "/usr/lib/python3/dist-packages"
#def touch(path):
   # with open(path, 'a'):
      #  os.utime(path, None)
#try:
  #  touch(DISTPACKAGES+"/__init__.py")
#except PermissionError:
  #  print("unable to touch __init__.py due to not having admin rights(this is intended)")
#importlib.import_module("/usr/lib/python3/dist-packages", package='libcamera')

from picamera2 import Picamera2




YAML = 'exports/data.yaml'
MODEL = 'exports/best.onnx'
SHAPE = (640,480)

class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, confidence_thres, iou_thres):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        #self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        self.classes = yaml_load(check_yaml('coco128.yaml'))['names']

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f'{self.classes[class_id]}: {score:.2f}'

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        self.img = self.input_array
        # Read the input image using OpenCV
        if not isinstance(self.img, np.ndarray):
            self.img = cv2.imread(self.input_image)
        
        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        #img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        #img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(self.img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return input_image

    def main(self):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # Create an inference session using the ONNX model and specify execution providers
        session = ort.InferenceSession(self.onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # Get the model inputs
        model_inputs = session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        print(input_shape)

        # Preprocess the image data
        img_data = self.preprocess()

        # Run inference using the preprocessed image data
        outputs = session.run(None, {model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(self.img, outputs)  # output image
    
    def set_image(self, img):
        self.input_array = img
        return


if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='yolov8n.onnx', help='Input your ONNX model.')
    # parser.add_argument('--img', type=str, default=str(ASSETS / 'bus.jpg'), help='Path to input image.')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    args = parser.parse_args()

    # Check the requirements and select the appropriate backend (CPU or GPU)
    #check_requirements('onnxruntime-gpu' if torch.cuda.is_available() else 'onnxruntime')
    check_requirements('onnxruntime')

    #model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(MODEL)
    model = MODEL

    # Create an instance of the YOLOv8 class with the specified arguments
    detection = YOLOv8(model, args.conf_thres, args.iou_thres)


    picam = Picamera2()

    config = {
        'size' : (640,640),
        'format': 'RGB888'
    }
    config = picam.create_still_configuration(main=config)
    picam.configure(config)

    picam.start()
    time.sleep(1)


    #cap = cv2.VideoCapture(0)
    while True:
        #ret, frame = cap.read()
        #picam.start()
        # Capture frame-by-frame
        array = picam.capture_array()
        # if frame is read correctly ret is True
        # if not ret:
        #     print("Can't receive frame (stream end?). Exiting ...")
        #     break
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform object detection and obtain the output image
        detection.set_image(array)
        output_image = detection.main()

        # Display the resulting frame
        #cv2.imshow('frame', output_image)
        if cv2.waitKey(1) == ord('q'):
            break
        

        # Display the output image in a window
        cv2.namedWindow('Output', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Output', output_image)

        # Wait for a key press to exit

        #picam.stop()

    #cap.release()
    picam.close()
    cv2.destroyAllWindows()