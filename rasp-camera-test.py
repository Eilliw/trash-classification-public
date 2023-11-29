import sys
import importlib

def apt_importer(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

pykms = apt_importer('pykms',"/usr/lib/python3/dist-packages/pykms/__init__.py")
libcamera = apt_importer('libcamera', '/usr/lib/python3/dist-packages/libcamera/__init__.py')
picamera2 = apt_importer('picamera2', '/usr/lib/python3/dist-packages/picamera2/__init__.py')

from picamera2 import Picamera2
import time
import cv2


picam2 = Picamera2()
time.sleep(1)
config = picam2.create_still_configuration(main= {"size": (1024, 1024)}, lores = {"size": (480, 320)}, display = "lores", buffer_count = 3, queue = False)
picam2.configure(config)

picam2.set_controls({"ExposureTime": 10000, "AnalogueGain": 5}) #Shutter time and analogue signal boost

picam2.start(show_preview=True)
time.sleep(10)
picam2.stop()
picam2.close()