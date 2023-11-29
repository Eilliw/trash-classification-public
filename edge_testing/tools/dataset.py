from roboflow import Roboflow
import os
import glob
import json



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
    def upload_dirs(self, dirs: list[str], ext=".jpg", annotations=None):
        for dir in dirs:
            clean_dir = dir.split("/")[-1]
            image_glob = glob.glob(dir + '/*' + ext)
            for img in image_glob:
                self.project.upload(img, annotation_path=clean_dir)
        return

if __name__ == "__main__":
    dataset = PiCamDataset()
    dataset.download_latest()
    dataset.upload_dirs(["lib/test_collection/Recycle","lib/test_collection/Trash"])