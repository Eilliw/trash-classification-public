from roboflow import Roboflow
import os
import glob
import json
from PIL import Image
from  datetime import datetime, timezone


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

class PiCamDataset():
    def __init__(self) -> None:
        self.rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
        self.workspace_name = "trashclassification-tayqe"
        self.project_name = "trash-vs-recycling-pi-cam"
        self.project = self.rf.workspace(self.workspace_name).project(self.project_name)
    
    def generate_latest(self):
        settings = {
            'agumentation':{},
            "preprocessing":{"resize":{"width":224, "height":224, "format": "Fit within"}}
        }
        gen = self.project.generate_version(settings)
        return gen
    def download_latest(self):
        versions = self.project.versions()
        self.dataset = self.project.version(len(versions)).download("folder",location="edge_testing/temp/latest_dataset",overwrite=True)
        #print(latest)
        
    def generate_annotations(self, dirs: list[str]):
        annotations = {}
        for dir in dirs:
            clean_dir = dir.split("/")[-1]
            annotations[clean_dir] = []
            image_glob = glob.glob(dir + '/*' + ".jpg")
            for img in image_glob:
                annotations[clean_dir].append(img)
        #seralize json
        json_object = json.dumps(annotations)
        json_path = "edge_testing/temp/annotations.json"
        with open(json_path, "w") as outfile:
            outfile.write(json_object)
        return json_path
    def upload_dir(self, dir, tags=[], ext=".jpg", with_annotations=False, batch_name  = None, delete=False):
        clean_dir = dir.split("/")[-1]
        image_glob = glob.glob(dir + '/*' + ext)
        for img in image_glob:
            if with_annotations==True:
                self.project.upload(img, annotation_path=clean_dir,  tag_names=tags, batch_name=batch_name)
            else:
                self.project.upload(img, tag_names=tags, batch_name=batch_name)
            if delete:
                os.remove(img)
    def upload_dirs(self, dirs: list[str], ext=".jpg", with_annotations=False):
        for dir in dirs:
            self.upload_dir(dir, with_annotations=with_annotations)
        return
    def collection_upload(self, path="lib/image_collection", delete=False):
        batch_name_datetime = datetime.now(timezone.utc).strftime("%y-%m-%d_%H:%M_%Z-auto_upload")
        self.upload_dir(path, tags=["Auto_upload", "unlabeled"], with_annotations=False, batch_name=batch_name_datetime, delete=delete)
    
    def save_to_local(self, img: Image.Image, path, path_pattern, sequential=True):
        cwd = os.getcwd()
        path  = cwd+"/"+path
        if sequential:
            if not os.path.exists(path):
                os.makedirs(path)
            next_name = next_path(path+"/"+path_pattern)
            try:
                img.save(next_name)
                print("saved image to "+next_name)
            except:
                raise
        else:
            if not os.path.exists(path):
                os.makedirs(path)
                img_name = path+"/"+datetime.now(timezone.utc).strftime("%y-%m-%d_%H:%M:%S_%Z.jpg")
                try:
                    img.save(img_name)
                    print("saved image to "+img_name)
                except:
                    print("couldn't save image")
    def get_latest_dataset_version(self):
        versions = self.project.versions()
        dataset = self.project.version(len(versions))
        return dataset
    def set_dataset_weights(self, weights_dir, version=-1, model_type = "yolov8-cls"):
        model_type = model_type
        if version == -1:
            dataset = self.get_latest_dataset_version()
            dataset.deploy(model_type, weights_dir)
        else:
            dataset = self.project.version(version)
            dataset.deploy(model_type, weights_dir)
        

if __name__ == "__main__":
    dataset = PiCamDataset()
    dataset.download_latest()
    #dataset.upload_dirs(["lib/test_collection/Recycle","lib/test_collection/Trash"])
    dataset.set_dataset_weights("trash-classification/3")
