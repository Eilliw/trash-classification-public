from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk
import numpy as np
import cv2
# from dataset import PiCamDataset
from  threading import Event

class TestingGUI:
    def __init__(self, dataset, image_flag : Event) -> None:
        self.root = Tk()
        self.root.title("Testing gui")
        self.root.geometry("360x360")

        self.dataset = dataset
        self.image_flag = image_flag

        self.exit_flag = Event()

        self.canvas = Canvas(self.root, height=224, width=224, bg='gray')
        self.canvas.grid(row=0, column=1, rowspan=4, columnspan=4)

        self.button_style = Style()
        self.button_style.configure("DF.TButton", foreground="white", background="grey")
        
        self.pressed_button_style = Style()
        self.pressed_button_style.configure("Pressed.TButton",  foreground="red", background='red')

        self.label_style = Style()
        self.label_style.configure("DF.TLabel", foreground="white", background="grey")

        self.buttons_labels()
        self.image_flag.set()

    
    def buttons_labels(self):
        self.btn_label_text = "Click the button based on what the image is"
        self.btn_label = Label(self.root, text=self.btn_label_text)
        self.btn_label.grid(row=5, column = 0, columnspan=4)

        self.classification_label_text = 'Classification: '
        self.classification_label =  Label(self.root, text=self.classification_label_text)
        self.classification_label.grid(row=6, column=0, columnspan=4)

        self.trash_button = Button(self.root, text="Trash",style="DF.TButton", command=self._trash_button_callback)
        self.trash_button.grid(row=1, column=0)

        self.recycle_button =  Button(self.root, text="Recycle", style="DF.TButton", command=self._recycle_button_callback)
        self.recycle_button.grid(row=2, column=0)

        self.upload_button = Button(self.root, text="Upload", style="DF.TButton" ,command=self._upload_button_callback)
        self.upload_button.grid(row=0, column=0)

        self.skip_button = Button(self.root, text="Skip", style='DF.TButton', command=self._skip_button_callback)
        self.skip_button.grid(row=4, column=0)


    def set_canvas_img(self, img :  np.ndarray):

        if isinstance(img, np.ndarray):
            res = cv2.resize(img, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
            self.canvas_img = ImageTk.PhotoImage(image=Image.fromarray(res))
            self.img = Image.fromarray(res)
            self.canvas.create_image(0,0, anchor = 'nw', image=self.canvas_img)
            # acquire condtion lock
            
        else:
            raise TypeError("needs  to be ndarray")
        self.image_flag.clear()
        self.buttons_ready()
    def set_classification(self, classification: tuple[float, str]):
        self.classification_label.configure(text=self.classification_label_text+classification[1]+"-"+str(classification[0]))
    def set_button_state(self, state,  style=None):
        #state  must be normal  or  disabled
        for btn in [self.trash_button, self.recycle_button,  self.upload_button, self.skip_button]:
            btn['state']=state
            if style != None:
                btn.configure(style=style)
    def after_button_press(self):
        self.set_button_state("disabled", style="Pressed.TButton")
        self.btn_label.configure(text="Waiting for image and classification to load")
        print("notifying threads")
        #notifys thread waiting on this condition
        self.image_flag.set()


    def buttons_ready(self):
        self.set_button_state("normal", style="DF.TButton")


    def _on_close(self):
        self.dataset.upload_dirs(['lib/test_collection/Recycle', "lib/test_collection/Trash"], with_annotations=True)
        
        #cleanup flags & let thread die by itself
        self.image_flag.set()
        self.exit_flag.set()
        self.root.destroy()


    def _trash_button_callback(self):
        self.btn_label.configure(text="Click the button based on what the image is")
        self.dataset.save_to_local(self.img, "lib/test_collection/Trash", "Trash-%s.jpg")
        self.after_button_press()

    def _recycle_button_callback(self):
        self.btn_label.configure(text="Click the button based on what the image is")
        self.dataset.save_to_local(self.img, "lib/test_collection/Recycle", "Recycle-%s.jpg")
        self.after_button_press()

    def _upload_button_callback(self):
        self.after_button_press()
        self.dataset.upload_dirs(["lib/test_collection/Recycle","lib/test_collection/Trash"])
    
    def _skip_button_callback(self):
        self.after_button_press()

if __name__ == "__main__":

    dataset = PiCamDataset()
    window = TestingGUI(dataset)

    image = Image.open("lib/test_collection/Trash/IMG_4283 Large.jpg")
    window.set_canvas_img(np.asarray(image))


    window.root.protocol("WM_DELETE_WINDOW", window._on_close)
    window.root.mainloop()